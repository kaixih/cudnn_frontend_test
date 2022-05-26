#include <cudnn_frontend.h>

#include "cudnn_frontend_utils.h"

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetToyGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
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

  struct GenericOp {
    std::optional<std::string> name;
    std::optional<cudnn_frontend::Tensor*> desc;
    // std::optional<cudnn_frontend::Operation&> op;
    GenericOp() {}
    // GenericOp(const std::string& n) : name(n) {}
    GenericOp(const char* c) : name(c) {}
    GenericOp(cudnn_frontend::Tensor* t) : desc(t) {}
  };

  struct Node {
    std::string op_name;
    cudnn_frontend::BackendDescriptor &desc;
    std::vector<double> scales;
    std::vector<GenericOp> inputs;
    std::vector<GenericOp> outputs;
  };
  std::unordered_map<std::string, int> test = {{"f", 12}, {"x", 32}};
  std::cout << test.at("f") << std::endl;
  std::cout << test.at("x") << std::endl;

  std::vector<Node> nodes = {
    {"convolution", conv_desc, {1., 0.}, {&tensor_x, &tensor_w}, {}},
    {"add", add_desc, {1., 0.}, {"convolution", {}, &tensor_z}, {}},
    {"bias_add", bias_add_desc, {}, {"add", {}, &tensor_b}, {}},
    {"relu", act_desc, {}, {"bias_add"}, {&tensor_y}}
  };

  // std::vector<Node> nodes1 = {
    // {"convolution", conv_desc, {1., 0.}, {&tensor_x, &tensor_w}, {&tensor_y}}
  // };

  int64_t reserved_uid = 1024;
  std::unordered_map<std::string, cudnn_frontend::Tensor> output_map;
  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].outputs.size() == 0) {
      std::cout << "Found empty output " << nodes[i].op_name << std::endl;
      ASSIGN_OR_RETURN(auto tensor_output,
                       CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                         opts.num_dims + 2, reserved_uid++, accumulator_type,
                                         /*is_virtual=*/true),
                       "Failed to build virtual tensor for " + nodes[i].op_name);
      output_map.emplace(std::make_pair(nodes[i].op_name, std::move(tensor_output)));
    }
  }
  std::cout << output_map.size() << std::endl;

  std::vector<cudnn_frontend::Operation> built_ops;
  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].op_name == "convolution") {
      cudnnBackendDescriptorType_t conv_kind =
          GetCudnnConvolutionType(opts.conv_kind);
      cudnn_frontend::Tensor* output_tensor;
      if (nodes[i].outputs.size() == 0) {
        // auto a = output_map.find("convolution");
        // if (a!= output_map.end()) {
          // output_tensor = &(a->second);
        // }
        // Don't use operator[] since it is non-const and will try to insert new
        // Tensor obj.
        output_tensor = &(output_map.at("convolution"));
      } else {
        output_tensor = nodes[i].outputs[0].desc.value();
      }
      auto conv_op = cudnn_frontend::OperationBuilder(conv_kind)
                     .setxDesc(*nodes[i].inputs[0].desc.value())
                     .setyDesc(*output_tensor)
                     .setwDesc(*nodes[i].inputs[1].desc.value())
                     .setcDesc(*(cudnn_frontend::ConvDesc*)(&nodes[i].desc))
                     .setAlpha(nodes[i].scales[0])
                     .setBeta(nodes[i].scales[1])
                     .build();
      RETURN_MSG_IF_CUDNN_ERROR(conv_op);
      built_ops.emplace_back(std::move(conv_op));
    } else if (nodes[i].op_name == "add") {
      cudnn_frontend::Tensor* output_tensor;
      if (nodes[i].outputs.size() == 0) {
        output_tensor = &(output_map.at("add"));
      } else {
        output_tensor = nodes[i].outputs[0].desc.value();
      }
      auto add_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(output_map.at(nodes[i].inputs[0].name.value())) // TODO
                    .setbDesc(*nodes[i].inputs[2].desc.value())
                    .setyDesc(*output_tensor)
                    .setpwDesc(*(cudnn_frontend::PointWiseDesc*)(&nodes[i].desc))
                    .setAlpha(nodes[i].scales[0])
                    .setAlpha2(nodes[i].scales[1])
                    .build();
      RETURN_MSG_IF_CUDNN_ERROR(add_op);
      built_ops.emplace_back(std::move(add_op));
    } else if (nodes[i].op_name == "bias_add") {
      cudnn_frontend::Tensor* output_tensor;
      if (nodes[i].outputs.size() == 0) {
        output_tensor = &(output_map.at("bias_add"));
      } else {
        output_tensor = nodes[i].outputs[0].desc.value();
      }
      auto add_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(output_map.at(nodes[i].inputs[0].name.value())) // TODO
                    .setbDesc(*nodes[i].inputs[2].desc.value())
                    .setyDesc(*output_tensor)
                    .setpwDesc(*(cudnn_frontend::PointWiseDesc*)(&nodes[i].desc))
                    .build();
      RETURN_MSG_IF_CUDNN_ERROR(add_op);
      built_ops.emplace_back(std::move(add_op));
    } else if (nodes[i].op_name == "relu") {
      cudnn_frontend::Tensor* output_tensor;
      if (nodes[i].outputs.size() == 0) {
        output_tensor = &(output_map.at("relu"));
      } else {
        std::cout << "relu uses real output\n";
        output_tensor = nodes[i].outputs[0].desc.value();
      }
      auto add_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(output_map.at(nodes[i].inputs[0].name.value())) // TODO
                    .setyDesc(*output_tensor)
                    .setpwDesc(*(cudnn_frontend::PointWiseDesc*)(&nodes[i].desc))
                    .build();
      RETURN_MSG_IF_CUDNN_ERROR(add_op);
      built_ops.emplace_back(std::move(add_op));
    }
  }
  std::cout << built_ops.size() << std::endl;
  std::array<cudnn_frontend::Operation const*, 4> ops = {
    &built_ops[0],
    &built_ops[1],
    &built_ops[2],
    &built_ops[3]};

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(op_graph);

  return std::unique_ptr<cudnn_frontend::OperationGraph>(
      new cudnn_frontend::OperationGraph(std::move(op_graph)));
}


