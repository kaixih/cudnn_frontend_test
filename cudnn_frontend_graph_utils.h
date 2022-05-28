#include <cudnn_frontend.h>

std::string CudnnStatusToString(cudnnStatus_t status);
int GetConvAccumulatorType(int data_type);
int GetConvActivationType(int data_type);
void CreateOpRunners(
    cudnnHandle_t& cudnn,
    std::unique_ptr<cudnn_frontend::OperationGraph> op_graph,
    std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>>* out_runners);
std::optional<cudnn_frontend::Tensor> CreateCudnnTensor(
    const int64_t* dims, const int64_t* strides, int n, int64_t uid, int dtype,
    bool is_virtual);
cudnnBackendDescriptorType_t GetCudnnConvolutionType(int64_t kind);

struct Edge {
  std::optional<std::string> tensor_name;
  std::optional<cudnn_frontend::Tensor*> tensor_ptr;
  Edge() {}
  Edge(const char* c) : tensor_name(c) {}
  Edge(cudnn_frontend::Tensor* t) : tensor_ptr(t) {}
};

struct Node {
  std::string op_name;
  int op_dtype;
  cudnn_frontend::BackendDescriptor& desc;
  std::vector<double> scales;
  std::unordered_map<std::string, Edge> edges;
};

std::optional<std::unique_ptr<cudnn_frontend::Operation>> GetConvolutionOp(
    Node& node, cudnnBackendDescriptorType_t& conv_kind,
    std::unordered_map<std::string, cudnn_frontend::Tensor*>& tensors) {
  auto op_builder = cudnn_frontend::OperationBuilder(conv_kind);

  cudnn_frontend::ConvDesc* conv_desc =
      reinterpret_cast<cudnn_frontend::ConvDesc*>(&node.desc);
  op_builder.setcDesc(*conv_desc);

  for (const auto& tensor : tensors) {
    if (tensor.first == node.op_name + ":x") {
      op_builder.setxDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":w") {
      op_builder.setwDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":y") {
      op_builder.setyDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":dy") {
      op_builder.setdyDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":dx") {
      op_builder.setdxDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":dw") {
      op_builder.setdwDesc(*tensor.second);
    }
  }

  if (node.scales.size() == 2) {
    op_builder.setAlpha(node.scales[0]);
    op_builder.setBeta(node.scales[1]);
  }

  auto op = op_builder.build();
  RETURN_MSG_IF_CUDNN_ERROR(op);
  return std::unique_ptr<cudnn_frontend::Operation>(
      new cudnn_frontend::Operation(std::move(op)));
}

std::optional<std::unique_ptr<cudnn_frontend::Operation>> GetPointwiseOp(
    Node& node,
    std::unordered_map<std::string, cudnn_frontend::Tensor*>& tensors) {
  auto op_builder = cudnn_frontend::OperationBuilder(
      CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);

  cudnn_frontend::PointWiseDesc* pw_desc =
      reinterpret_cast<cudnn_frontend::PointWiseDesc*>(&node.desc);
  op_builder.setpwDesc(*pw_desc);

  for (const auto& tensor : tensors) {
    if (tensor.first == node.op_name + ":x") {
      op_builder.setxDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":b") {
      op_builder.setbDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":y") {
      op_builder.setyDesc(*tensor.second);
    } else if (tensor.first == node.op_name + ":t") {
      op_builder.settDesc(*tensor.second);
    }
  }

  if (node.scales.size() == 2) {
    op_builder.setAlpha(node.scales[0]);
    op_builder.setAlpha2(node.scales[1]);
  }

  auto op = op_builder.build();
  RETURN_MSG_IF_CUDNN_ERROR(op);
  return std::unique_ptr<cudnn_frontend::Operation>(
      new cudnn_frontend::Operation(std::move(op)));
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> CreateOpGraph(
    ConvOpts& opts, cudnnHandle_t& cudnn, std::vector<Node> &nodes) {
  // We move the virtual tensors into the map to extend their liveness. Note,
  // don't use the vector since the reallocation will change the addresses of
  // the tensors.
  std::unordered_map<std::string, cudnn_frontend::Tensor> virtual_tensors;
  // All tensors are marked in "tensors" in the form of:
  //   "op_name:port" -> tensor address
  std::unordered_map<std::string, cudnn_frontend::Tensor*> tensors;
  int64_t reserved_uid = 1024;
  // Put virtual (connecting) tensors and given (end) tensors to "tensors". In
  // this step the virtual tensors can only be marked in the fanout sides.
  for (int i = 0; i < nodes.size(); i++) {
    for (const auto& edge : nodes[i].edges) {
      std::string tag = nodes[i].op_name + ":" + edge.first;
      auto tensor_name_or = edge.second.tensor_name;
      auto tensor_ptr_or = edge.second.tensor_ptr;
      if (tensor_name_or.has_value() && tensor_name_or.value() == "") {
        // The virtual tensor dtype is determined by the op dtype.
        ASSIGN_OR_RETURN(
            auto tensor_output,
            CreateCudnnTensor(opts.output_dims, opts.output_strides,
                              opts.num_dims + 2, reserved_uid++,
                              nodes[i].op_dtype, /*is_virtual=*/true),
            "Failed to build the virtual tensor for " + nodes[i].op_name);

        virtual_tensors.insert({tag, std::move(tensor_output)});
        tensors[tag] = &virtual_tensors.at(tag);
      } else if (tensor_ptr_or.has_value()) {
        tensors[tag] = tensor_ptr_or.value();
      }
    }
  }
  // Put virtual tensors to "tensors". In this step the virtual tensors can be
  // marked in the fanin sides.
  for (int i = 0; i < nodes.size(); i++) {
    for (auto& edge : nodes[i].edges) {
      auto tensor_name_or = edge.second.tensor_name;
      if (tensor_name_or.has_value() && tensor_name_or.value() != "") {
        auto found = tensors.find(tensor_name_or.value());
        if (found == tensors.end()) {
          std::cout << "!!! Graph error: cannot find " << tensor_name_or.value()
                    << std::endl;
          return {};
        }
        tensors[nodes[i].op_name + ":" + edge.first] = found->second;
      }
    }
  }

  std::vector<cudnn_frontend::Operation> built_ops;
  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].op_name.find("convolution") != std::string::npos) {
      int op_index;
      if (nodes[i].op_name == "convolution") {
        op_index = 0;
      } else if (nodes[i].op_name == "convolution_bwd_filter") {
        op_index = 1;
      } else if (nodes[i].op_name == "convolution_bwd_input") {
        op_index = 2;
      }
      cudnnBackendDescriptorType_t conv_kind =
          GetCudnnConvolutionType(op_index);
      ASSIGN_OR_RETURN(auto op, GetConvolutionOp(nodes[i], conv_kind, tensors),
                       "Failed to build op" + nodes[i].op_name);
      built_ops.emplace_back(std::move(*op));
    } else {
      ASSIGN_OR_RETURN(auto op, GetPointwiseOp(nodes[i], tensors),
                       "Failed to build op" + nodes[i].op_name);
      built_ops.emplace_back(std::move(*op));
    }
  }

  std::vector<const cudnn_frontend::Operation*> ops;
  for (int i = 0; i < built_ops.size(); i++) {
    ops.push_back(&built_ops[i]);
  }

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(op_graph);

  return std::unique_ptr<cudnn_frontend::OperationGraph>(
      new cudnn_frontend::OperationGraph(std::move(op_graph)));
}


