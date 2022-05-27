#include <cudnn_frontend.h>

#include "cudnn_frontend_utils.h"

struct Edge {
  std::optional<std::string> name;
  std::optional<cudnn_frontend::Tensor*> desc;
  Edge() {}
  Edge(const char* c) : name(c) {}
  Edge(cudnn_frontend::Tensor* t) : desc(t) {}
};

struct Node {
  std::string op_name;
  cudnn_frontend::BackendDescriptor& desc;
  std::vector<double> scales;
  std::unordered_map<std::string, Edge> inputs;
  std::unordered_map<std::string, Edge> outputs;
};

struct ProcessedNode {
  std::string op_name;
  cudnn_frontend::BackendDescriptor* desc;
  std::vector<double> scales;
  std::unordered_map<std::string, cudnn_frontend::Tensor*> in_out;
};

std::optional<std::unique_ptr<cudnn_frontend::Operation>> GetPointwiseOp(
    ProcessedNode& processed_node) {
  auto op_builder = cudnn_frontend::OperationBuilder(
      CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);

  cudnn_frontend::PointWiseDesc* pw_desc =
      reinterpret_cast<cudnn_frontend::PointWiseDesc*>(processed_node.desc);
  op_builder.setpwDesc(*pw_desc);

  for (const auto& item : processed_node.in_out) {
    if (item.first == "x") {
      op_builder.setxDesc(*item.second);
    } else if (item.first == "b") {
      op_builder.setbDesc(*item.second);
    } else if (item.first == "y") {
      op_builder.setyDesc(*item.second);
    } else {
      std::cout << "!!! tensor tag: " << item.first << " is not supported" << std::endl;
      return {};
    }
  }

  if (processed_node.scales.size() == 2) {
    op_builder.setAlpha(processed_node.scales[0]);
    op_builder.setAlpha2(processed_node.scales[1]);
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

  std::vector<Node> nodes = {
      {"convolution",
       conv_desc,
       {1., 0.},
       {{"x", &tensor_x}, {"w", &tensor_w}},
       {}},
      {"add", add_desc, {1., 0.}, {{"x", "convolution"}, {"b", &tensor_z}}, {}},
      {"bias_add", bias_add_desc, {}, {{"x", "add"}, {"b", &tensor_b}}, {}},
      {"relu", act_desc, {}, {{"x", "bias_add"}}, {{"y", &tensor_y}}}};

  // std::vector<Node> nodes1 = {
  // {"convolution", conv_desc, {1., 0.}, {&tensor_x, &tensor_w}, {&tensor_y}}
  // };

  int64_t reserved_uid = 1024;
  std::unordered_map<std::string, cudnn_frontend::Tensor> op_output_map;
  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].outputs.size() == 0) {
      std::cout << "Creating a virtual output tensor for " << nodes[i].op_name
                << std::endl;
      int output_dtype = nodes[i].op_name == "convolution" ? accumulator_type
                                                           : activation_type;
      ASSIGN_OR_RETURN(
          auto tensor_output,
          CreateCudnnTensor(opts.output_dims, opts.output_strides,
                            opts.num_dims + 2, reserved_uid++, output_dtype,
                            /*is_virtual=*/true),
          "Failed to build virtual tensor for " + nodes[i].op_name);

      op_output_map.emplace(
          std::make_pair(nodes[i].op_name, std::move(tensor_output)));
    }
  }
  for (auto& item : op_output_map) {
    std::cout << item.first << std::endl;
  }

  std::cout << "Stage 1" << std::endl;
  std::vector<ProcessedNode> processed_nodes;
  for (int i = 0; i < nodes.size(); i++) {
    ProcessedNode processed_node;
    processed_node.op_name = nodes[i].op_name;
    processed_node.desc = &nodes[i].desc;
    processed_node.scales = nodes[i].scales;
    std::cout << nodes[i].op_name << ": ";
    // std::unordered_map<std::string, cudnn_frontend::Tensor*> in_out;
    for (auto& input : nodes[i].inputs) {
      cudnn_frontend::Tensor* t;
      if (input.second.name.has_value()) {
        t = &(op_output_map.at(input.second.name.value()));
      } else if (input.second.desc.has_value()) {
        t = input.second.desc.value();
      }
      processed_node.in_out[input.first] = t;
      std::cout << input.first << ", ";
    }

    for (auto& output : nodes[i].outputs) {
      cudnn_frontend::Tensor* t;
      if (output.second.desc.has_value()) {
        t = output.second.desc.value();
      }
      processed_node.in_out[output.first] = t;
      std::cout << output.first << ", ";
    }

    auto got_virtual_output = op_output_map.find(nodes[i].op_name);
    if (got_virtual_output != op_output_map.end()) {
      processed_node.in_out["y"] = &(got_virtual_output->second);
      std::cout << "y, ";
    }
    std::cout << std::endl;
    processed_nodes.push_back(std::move(processed_node));
  }

  std::vector<cudnn_frontend::Operation> built_ops;
  for (int i = 0; i < processed_nodes.size(); i++) {
    if (processed_nodes[i].op_name == "convolution") {
      cudnnBackendDescriptorType_t conv_kind =
          GetCudnnConvolutionType(opts.conv_kind);

      cudnn_frontend::Tensor* y_tensor = processed_nodes[i].in_out.at("y");
      cudnn_frontend::Tensor* x_tensor = processed_nodes[i].in_out.at("x");
      cudnn_frontend::Tensor* w_tensor = processed_nodes[i].in_out.at("w");
      auto conv_op =
          cudnn_frontend::OperationBuilder(conv_kind)
              .setxDesc(*x_tensor)
              .setyDesc(*y_tensor)
              .setwDesc(*w_tensor)
              .setcDesc(*(cudnn_frontend::ConvDesc*)(processed_nodes[i].desc))
              .setAlpha(processed_nodes[i].scales[0])
              .setBeta(processed_nodes[i].scales[1])
              .build();
      RETURN_MSG_IF_CUDNN_ERROR(conv_op);
      built_ops.emplace_back(std::move(conv_op));
    } else {
      std::cout << "processing " << processed_nodes[i].op_name << std::endl;
      ASSIGN_OR_RETURN(auto op, GetPointwiseOp(processed_nodes[i]),
                       "Failed to build op" + processed_nodes[i].op_name);
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
