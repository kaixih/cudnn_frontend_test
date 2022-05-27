#include <cudnn_frontend.h>

#include "cudnn_frontend_utils.h"

struct Edge {
  std::optional<std::string> tensor_name;
  std::optional<cudnn_frontend::Tensor*> tensor_ptr;
  Edge() {}
  Edge(const char* c) : tensor_name(c) {}
  Edge(cudnn_frontend::Tensor* t) : tensor_ptr(t) {}
};

struct Node {
  std::string op_name;
  cudnn_frontend::BackendDescriptor& desc;
  std::vector<double> scales;
  std::unordered_map<std::string, Edge> edges;
};

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

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> GetToyGraph(
    ConvOpts& opts, cudnnHandle_t& cudnn) {
  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'z', opts.data_type),
                   "Failed to build tensor z");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims + 2, 'b', opts.data_type),
                   "Failed to build tensor b");

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  int activation_type = GetConvActivationType(opts.data_type);
  cudnnDataType_t cudnn_convolution_type = ToCudnnDataType(accumulator_type);
  cudnnDataType_t cudnn_activation_type = ToCudnnDataType(activation_type);
  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputePrecision(cudnn_convolution_type)
                       .setMathMode(conv_mode)
                       .setNDims(conv_dim)
                       .setStrides(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  auto add_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ADD)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(add_desc);

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(cudnn_activation_type)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_desc);

  auto act_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_RELU_FWD)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(act_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", conv_desc, {1., 0.},
         /*edges=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"add", add_desc, {1., 0.},
         /*edges=*/{{"x", "convolution:y"}, {"b", &tensor_z}, {"y", ""}}},
      {"bias_add", bias_add_desc, {},
         /*edges=*/{{"x", "add:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"relu", act_desc, {},
         /*edges=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  // std::vector<Node> nodes1 = {
  // {"convolution", conv_desc, {1., 0.}, {&tensor_x, &tensor_w}, {&tensor_y}}
  // };

  // We move the virtual tensors into the map to extend their liveness. Note,
  // don't use the vector since the reallocation will change the addresses of
  // the tensors.
  std::unordered_map<std::string, cudnn_frontend::Tensor> virtual_tensors;
  // All tensors are marked in "tensors" in the form of:
  //   "op_name:specifier" -> tensor address
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
        int output_dtype = nodes[i].op_name == "convolution" ? accumulator_type
                                                             : activation_type;
        ASSIGN_OR_RETURN(
            auto tensor_output,
            CreateCudnnTensor(opts.output_dims, opts.output_strides,
                              opts.num_dims + 2, reserved_uid++, output_dtype,
                              /*is_virtual=*/true),
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
    if (nodes[i].op_name == "convolution") {
      cudnnBackendDescriptorType_t conv_kind =
          GetCudnnConvolutionType(opts.conv_kind);

      cudnn_frontend::Tensor* y_tensor = tensors.at("convolution:y");
      cudnn_frontend::Tensor* x_tensor = tensors.at("convolution:x");
      cudnn_frontend::Tensor* w_tensor = tensors.at("convolution:w");
      auto conv_op_builder =
          cudnn_frontend::OperationBuilder(conv_kind);
      conv_op_builder.setxDesc(*x_tensor);
      conv_op_builder.setyDesc(*y_tensor);
      conv_op_builder.setwDesc(*w_tensor);
      conv_op_builder.setcDesc(*(cudnn_frontend::ConvDesc*)(&nodes[i].desc));
      conv_op_builder.setAlpha(nodes[i].scales[0]);
      conv_op_builder.setBeta(nodes[i].scales[1]);
      auto conv_op = conv_op_builder.build();

      RETURN_MSG_IF_CUDNN_ERROR(conv_op);
      built_ops.emplace_back(std::move(conv_op));
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
