#include <cudnn_frontend.h>

#include "cudnn_frontend_graph_utils.h"
#include "cudnn_frontend_utils.h"

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

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  int activation_type = GetConvActivationType(opts.data_type);
  
  auto conv_mode = CUDNN_CROSS_CORRELATION;
  int conv_dim = opts.num_dims;

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

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(cudnn_activation_type)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_desc);

  auto elu_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ELU_FWD)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(elu_desc);

  // clang-format off
  std::vector<Node> nodes = {
      {"convolution", accumulator_type, conv_desc, {1., 0.},
         /*edges=*/{{"x", &tensor_x}, {"w", &tensor_w}, {"y", ""}}},
      {"bias_add", accumulator_type, bias_add_desc, {},
         /*edges=*/{{"x", "convolution:y"}, {"b", &tensor_b}, {"y", ""}}},
      {"elu", activation_type, elu_desc, {},
         /*edges=*/{{"x", "bias_add:y"}, {"y", &tensor_y}}}};
  // clang-format on

  return CreateOpGraph(cudnn, nodes);
}
