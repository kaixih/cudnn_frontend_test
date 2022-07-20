#include <cudnn_frontend.h>

#include "cudnn_frontend_graph_utils.h"
#include "cudnn_frontend_utils.h"

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetMatMulBiasTanhGraph(MatMulOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // MatMul + BiasAdd + Tanh. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // MatMul : input: tensor_a, tensor_b;      output: tensor_matmul (virtual)
  // BiasAdd: input: tensor_matmul, tensor_z; output: tensor_bias (virtual)
  // Tanh   : input: tensor_bias;             output: tensor_c

  ASSIGN_OR_RETURN(auto tensor_a,
                   CreateCudnnTensor(opts.input0_dims, opts.input0_strides,
                                     opts.num_dims, 'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.input1_dims, opts.input1_strides,
                                     opts.num_dims, 'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims, 'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  int activation_type = GetConvActivationType(opts.data_type);
  
  cudnnDataType_t cudnn_matmul_type = ToCudnnDataType(accumulator_type);
  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setMathPrecision(cudnn_matmul_type)
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
                   CreateCudnnTensor(opts.input0_dims, opts.input0_strides,
                                     opts.num_dims, 'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.input1_dims, opts.input1_strides,
                                     opts.num_dims, 'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims, 'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  int activation_type = GetConvActivationType(opts.data_type);
  
  cudnnDataType_t cudnn_matmul_type = ToCudnnDataType(accumulator_type);
  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setMathPrecision(cudnn_matmul_type)
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
GetMatMulBiasGeluGraph(MatMulOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // MatMul + BiasAdd + Tanh. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // MatMul : input: tensor_a, tensor_b;      output: tensor_matmul (virtual)
  // BiasAdd: input: tensor_matmul; output: tensor_bias (virtual)
  // Gelu:    input: tensor_bias;   output: tensor_c

  ASSIGN_OR_RETURN(auto tensor_a,
                   CreateCudnnTensor(opts.input0_dims, opts.input0_strides,
                                     opts.num_dims, 'a', opts.data_type),
                   "Failed to build tensor a");

  ASSIGN_OR_RETURN(auto tensor_b,
                   CreateCudnnTensor(opts.input1_dims, opts.input1_strides,
                                     opts.num_dims, 'b', opts.data_type),
                   "Failed to build tensor b");

  ASSIGN_OR_RETURN(auto tensor_c,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims, 'c', opts.data_type),
                   "Failed to build tensor c");

  ASSIGN_OR_RETURN(auto tensor_z,
                   CreateCudnnTensor(opts.bias_dims, opts.bias_strides,
                                     opts.num_dims, 'z', opts.data_type),
                   "Failed to build tensor z");

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  int activation_type = GetConvActivationType(opts.data_type);
  
  cudnnDataType_t cudnn_matmul_type = ToCudnnDataType(accumulator_type);
  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setMathPrecision(cudnn_matmul_type)
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
