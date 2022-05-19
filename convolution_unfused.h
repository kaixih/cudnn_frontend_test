#include <cudnn_frontend.h>

#include "cudnn_frontend_utils.h"

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetUnfusedConvGraph(ConvOpts& opts, cudnnHandle_t& cudnn) {
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
  auto accumulator_type =
      ToCudnnDataType(GetConvAccumulatorType(opts.data_type));

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setComputePrecision(accumulator_type)
                       .setMathMode(conv_mode)
                       .setNDims(conv_dim)
                       .setStrides(conv_dim, opts.strides)
                       .setPrePadding(conv_dim, opts.paddings)
                       .setPostPadding(conv_dim, opts.paddings)
                       .setDilation(conv_dim, opts.dilations)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  double alpha = 1.0;
  double beta = 0.0;

  cudnnBackendDescriptorType_t conv_kind =
      GetCudnnConvolutionType(opts.conv_kind);
  auto op = cudnn_frontend::OperationBuilder(conv_kind)
                .setxDesc(tensor_x)
                .setyDesc(tensor_y)
                .setwDesc(tensor_w)
                .setcDesc(conv_desc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
  RETURN_MSG_IF_CUDNN_ERROR(op);

  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(cudnn)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(opGraph);

  return std::unique_ptr<cudnn_frontend::OperationGraph>(
      new cudnn_frontend::OperationGraph(std::move(opGraph)));
}

