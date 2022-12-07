#include "graph_builder.h"

#include "graph_util.h"

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> GetConvFwdGraph(
    ConvOpts& opts, cudnnHandle_t& cudnn) {
  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;
  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", &tensor_y}}}};
  // clang-format on
  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBwdFilterGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;
  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution_bwd_filter", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"dw", &tensor_w}, {"dy", &tensor_y}}}};
  // clang-format on
  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBwdDataGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;
  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution_bwd_input", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"dx", &tensor_x}, {"w", &tensor_w}, {"dy", &tensor_y}}}};
  // clang-format on
  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvAddBiasReluGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // Conv + Add + BiasAdd + Relu. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;    output: tensor_conv (virtual)
  // Add    : input: tensor_conv, tensor_z; output: tensor_add (virtual)
  // BiasAdd: input: tensor_add, tensor_b;  output: tensor_bias (virtual)
  // Relu   : input: tensor_bias;           output: tensor_y

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

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"add", accumulator_type, nullptr, {1., 0.},
         /*ports=*/{{"x", "convolution:y"}, {"b", &tensor_z}, {"y", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         /*ports=*/{{"x", "add:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"relu", activation_type, nullptr, {},
         /*ports=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBiasEluGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // Conv + BiasAdd + Elu. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;     output: tensor_conv (virtual)
  // BiasAdd: input: tensor_conv, tensor_b;  output: tensor_bias (virtual)
  // Elu    : input: tensor_bias;            output: tensor_y

  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

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

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  auto elu_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ELU_FWD)
                      .setComputeType(activation_type)
                      .setEluAlpha(1.0)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(elu_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         /*ports=*/{{"x", "convolution:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"elu", activation_type, &elu_desc, {},
         /*ports=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBiasRelu6Graph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // Conv + BiasAdd + Relu6. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;    output: tensor_conv (virtual)
  // BiasAdd: input: tensor_conv, tensor_b; output: tensor_bias (virtual)
  // RELU6  : input: tensor_bias;           output: tensor_y

  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims + 2, 'b', opts.data_type),
                   "Failed to build tensor b");

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         /*ports=*/{{"x", "convolution:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"relu6", activation_type, nullptr, {},
         /*ports=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBiasLeakyReluGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // Conv + BiasAdd + LeakyRelu. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv      : input: tensor_x, tensor_w;    output: tensor_conv (virtual)
  // BiasAdd   : input: tensor_conv, tensor_b; output: tensor_bias (virtual)
  // LeakyRelu : input: tensor_bias;           output: tensor_y

  ASSIGN_OR_RETURN(auto tensor_x,
                   CreateCudnnTensor(opts.input_dims, opts.input_strides,
                                     opts.num_dims + 2, 'x', opts.data_type),
                   "Failed to build tensor x");

  ASSIGN_OR_RETURN(auto tensor_y,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'y', opts.data_type),
                   "Failed to build tensor y");

  ASSIGN_OR_RETURN(auto tensor_w,
                   CreateCudnnTensor(opts.filter_dims, opts.filter_strides,
                                     opts.num_dims + 2, 'w', opts.data_type),
                   "Failed to build tensor w");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims + 2, 'b', opts.data_type),
                   "Failed to build tensor b");

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputeType(accumulator_type)
                       .setMathMode(conv_mode)
                       .setSpatialDimCount(conv_dim)
                       .setSpatialStride(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  auto leakyrelu_desc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_RELU_FWD)
                            .setComputeType(activation_type)
                            .setReluLowerClipSlope(0.3)  // leaky relu
                            .build();
  RETURN_MSG_IF_CUDNN_ERROR(leakyrelu_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, &conv_desc, {1., 0.},
         /*ports=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         /*ports=*/{{"x", "convolution:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"leakyrelu", activation_type, &leakyrelu_desc, {},
         /*ports=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetMatMulBiasTanhGraph(MatMulOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // MatMul + BiasAdd + Tanh. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // MatMul : input: tensor_a, tensor_b;      output: tensor_matmul (virtual)
  // BiasAdd: input: tensor_matmul, tensor_z; output: tensor_bias (virtual)
  // Tanh   : input: tensor_bias;             output: tensor_c

  ASSIGN_OR_RETURN(auto tensor_a,
                   CreateCudnnTensor(opts.a_dims, opts.a_strides, opts.num_dims,
                                     'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.b_dims, opts.b_strides, opts.num_dims,
                                     'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.c_dims, opts.c_strides, opts.num_dims,
                                     'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setComputeType(accumulator_type)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(matmul_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"matmul", accumulator_type, &matmul_desc, {},
         {{"a", &tensor_a}, {"b", &tensor_b}, {"c", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         {{"x", "matmul:c"}, {"b", &tensor_z}, {"y", ""}}},
      {"tanh", activation_type, nullptr, {},
         {{"x", "bias_add:y"}, {"y", &tensor_c}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetMatMulBiasSigmoidGraph(MatMulOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // MatMul + BiasAdd + Tanh. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // MatMul : input: tensor_a, tensor_b;      output: tensor_matmul (virtual)
  // BiasAdd: input: tensor_matmul; output: tensor_bias (virtual)
  // Sigmoid: input: tensor_bias;   output: tensor_c

  ASSIGN_OR_RETURN(auto tensor_a,
                   CreateCudnnTensor(opts.a_dims, opts.a_strides, opts.num_dims,
                                     'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.b_dims, opts.b_strides, opts.num_dims,
                                     'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.c_dims, opts.c_strides, opts.num_dims,
                                     'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setComputeType(accumulator_type)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(matmul_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"matmul", accumulator_type, &matmul_desc, {},
         {{"a", &tensor_a}, {"b", &tensor_b}, {"c", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         {{"x", "matmul:c"}, {"b", &tensor_z}, {"y", ""}}},
      {"sigmoid", activation_type, nullptr, {},
         {{"x", "bias_add:y"}, {"y", &tensor_c}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetMatMulBiasGeluExactGraph(MatMulOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // MatMul + BiasAdd + Tanh. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // MatMul :   input: tensor_a, tensor_b; output: tensor_matmul (virtual)
  // BiasAdd:   input: tensor_matmul;      output: tensor_bias (virtual)
  // GeluExact: input: tensor_bias;        output: tensor_c

  ASSIGN_OR_RETURN(auto tensor_a,
                   CreateCudnnTensor(opts.a_dims, opts.a_strides, opts.num_dims,
                                     'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.b_dims, opts.b_strides, opts.num_dims,
                                     'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.c_dims, opts.c_strides, opts.num_dims,
                                     'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  auto accumulator_type = GetConvAccumulatorCudnnDataType(opts.data_type);
  auto activation_type = GetConvActivationCudnnDataType(opts.data_type);

  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setComputeType(accumulator_type)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(matmul_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"matmul", accumulator_type, &matmul_desc, {},
         {{"a", &tensor_a}, {"b", &tensor_b}, {"c", ""}}},
      {"bias_add", accumulator_type, nullptr, {},
         {{"x", "matmul:c"}, {"b", &tensor_z}, {"y", ""}}},
      {"gelu_exact", activation_type, nullptr, {},
         {{"x", "bias_add:y"}, {"y", &tensor_c}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}

void PrintGraphName(int graph_index) {
  printf(
      ">>>   graph_index (-graph_index <int>(+100 for matmul graphs)): %d\n",
      graph_index);
  auto graph_type = static_cast<GraphType>(graph_index);
  switch (graph_type) {
    case GraphType::ConvFwd:
      printf(">>>   graph_name: ConvFwdGraph\n");
      break;
    case GraphType::ConvBwdFilter:
      printf(">>>   graph_name: ConvBwdFilterGraph\n");
      break;
    case GraphType::ConvBwdData:
      printf(">>>   graph_name: ConvBwdDataGraph\n");
      break;
    case GraphType::ConvAddBiasRelu:
      printf(">>>   graph_name: ConvAddBiasReluGraph\n");
      break;
    case GraphType::ConvBiasElu:
      printf(">>>   graph_name: ConvBiasEluGraph\n");
      break;
    case GraphType::ConvBiasRelu6:
      printf(">>>   graph_name: ConvBiasRelu6Graph\n");
      break;
    case GraphType::ConvBiasLeakyRelu:
      printf(">>>   graph_name: ConvBiasLeakyReluGraph\n");
      break;
    case GraphType::MatMulBiasTanh:
      printf(">>>   graph_name: MatMulBiasTanhGraph\n");
      break;
    case GraphType::MatMulBiasSigmoid:
      printf(">>>   graph_name: MatMulBiasSigmoidGraph\n");
      break;
    case GraphType::MatMulBiasGeluExact:
      printf(">>>   graph_name: MatMulBiasGeluExactGraph\n");
      break;
    default:
      printf(RED "!!! Unsupported graph index: %d\n" RESET, graph_index);
  }
}

std::optional<ConvGraphBuilderFnPtr> GetConvGraphBuilderByIndex(
    int graph_index) {
  auto graph_type = static_cast<GraphType>(graph_index);
  switch (graph_type) {
    case GraphType::ConvFwd:
      return GetConvFwdGraph;
    case GraphType::ConvBwdFilter:
      return GetConvBwdFilterGraph;
    case GraphType::ConvBwdData:
      return GetConvBwdDataGraph;
    case GraphType::ConvAddBiasRelu:
      return GetConvAddBiasReluGraph;
    case GraphType::ConvBiasElu:
      return GetConvBiasEluGraph;
    case GraphType::ConvBiasRelu6:
      return GetConvBiasRelu6Graph;
    case GraphType::ConvBiasLeakyRelu:
      return GetConvBiasLeakyReluGraph;
    default:
      printf(RED "!!! Unsupported conv graph index: %d\n" RESET, graph_index);
      return {};
  }
}

std::optional<MatMulGraphBuilderFnPtr> GetMatMulGraphBuilderByIndex(
    int graph_index) {
  auto graph_type = static_cast<GraphType>(graph_index);
  switch (graph_type) {
    case GraphType::MatMulBiasTanh:
      return GetMatMulBiasTanhGraph;
    case GraphType::MatMulBiasSigmoid:
      return GetMatMulBiasSigmoidGraph;
    case GraphType::MatMulBiasGeluExact:
      return GetMatMulBiasGeluExactGraph;
    default:
      printf(RED "!!! Unsupported matmul graph index: %d\n" RESET, graph_index);
      return {};
  }
}
