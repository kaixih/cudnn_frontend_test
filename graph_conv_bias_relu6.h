#include <cudnn_frontend.h>

#include "cudnn_frontend_utils.h"

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetConvBiasRelu6Graph(ConvOpts& opts, cudnnHandle_t& cudnn) {
  // CUDNN fused operation supports the pattern in the form of
  // Conv + BiasAdd + Relu6. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;       output: tensor_conv (virtual)
  // BiasAdd: input: tensor_conv, tensor_b;    output: tensor_bias (virtual)
  // MIN    : input: tensor_bias, tensor_zero; output: tensor_min (virtual)
  // MAX    : input: tensor_min, tensor_six;   output: tensor_y

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

  std::vector<int64_t> scalar_dims(opts.num_dims + 2, 1);
  std::vector<int64_t> scalar_strides(opts.num_dims + 2, 1);
  ASSIGN_OR_RETURN(auto scalar_tensor_zero,
                   CreateCudnnTensor(scalar_dims.data(), scalar_strides.data(),
                                     opts.num_dims + 2, '0', opts.data_type),
                   "Failed to build tensor 0");
  ASSIGN_OR_RETURN(auto scalar_tensor_six,
                   CreateCudnnTensor(scalar_dims.data(), scalar_strides.data(),
                                     opts.num_dims + 2, '6', opts.data_type),
                   "Failed to build tensor 6");

  int accumulator_type = GetConvAccumulatorType(opts.data_type);
  ASSIGN_OR_RETURN(auto tensor_conv,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'C', accumulator_type,
                                     /*is_virtual=*/true),
                   "Failed to build tensor conv");

  int activation_type = GetConvActivationType(opts.data_type);
  ASSIGN_OR_RETURN(auto tensor_bias,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'B', activation_type,
                                     /*is_virtual=*/true),
                   "Failed to build tensor bias");
  ASSIGN_OR_RETURN(auto tensor_min,
                   CreateCudnnTensor(opts.output_dims, opts.output_strides,
                                     opts.num_dims + 2, 'M', activation_type,
                                     /*is_virtual=*/true),
                   "Failed to build tensor min");

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

  cudnnBackendDescriptorType_t conv_kind =
      GetCudnnConvolutionType(opts.conv_kind);
  auto conv_op = cudnn_frontend::OperationBuilder(conv_kind)
                     .setxDesc(tensor_x)
                     .setyDesc(tensor_conv)
                     .setwDesc(tensor_w)
                     .setcDesc(conv_desc)
                     .setAlpha(1.0f)
                     .setBeta(0.0f)
                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_op);

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(cudnn_activation_type)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_desc);

  auto bias_add_op = cudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(conv_op.getOutputTensor())
                         .setbDesc(tensor_b)
                         .setyDesc(tensor_bias)
                         .setpwDesc(bias_add_desc)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_op);

  auto min_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_MIN)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(min_desc);

  auto min_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(bias_add_op.getOutputTensor())
                    .setbDesc(scalar_tensor_zero)
                    .setyDesc(tensor_min)
                    .setpwDesc(min_desc)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(min_op);

  auto max_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_MAX)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(max_desc);

  auto max_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(bias_add_op.getOutputTensor())
                    .setbDesc(scalar_tensor_six)
                    .setyDesc(tensor_y)
                    .setpwDesc(max_desc)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(max_op);


  std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_add_op,
                                                         &min_op, &max_op};

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(op_graph);

  return std::unique_ptr<cudnn_frontend::OperationGraph>(
      new cudnn_frontend::OperationGraph(std::move(op_graph)));
}
