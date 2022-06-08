#include <cudnn.h>

#include <numeric>

#define checkCUDA(expression)                               \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess) {                            \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error from file: " << __FILE__  << " on "\
                << "line " << __LINE__ << ": "               \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define UNIQUE_VAR_NAME(x, y) UNIQUE_VAR_NAME_IMPL(x, y)
#define UNIQUE_VAR_NAME_IMPL(x, y) x##y

#define ASSIGN_OR_RETURN_IMPL(lhs, val_or, rexpr, err) \
  auto val_or = (rexpr);                               \
  if (!val_or.has_value()) {                           \
    std::cout << "!!! " << err << std::endl;           \
  }                                                    \
  lhs = std::move(val_or.value());

#define ASSIGN_OR_RETURN(lhs, rexpr, err) \
  ASSIGN_OR_RETURN_IMPL(lhs, UNIQUE_VAR_NAME(_val_or, __COUNTER__), rexpr, err)

#define RETURN_MSG_IF_CUDNN_ERROR(expr)                                 \
  do {                                                                  \
    cudnnStatus_t _status = (expr).get_status();                        \
    if (!(_status == CUDNN_STATUS_SUCCESS)) {                           \
      std::ostringstream oss;                                           \
      oss << CudnnStatusToString(_status) << "\nin " << __FILE__ << "(" \
          << __LINE__ << "): '" << #expr << "' " << (expr).get_error(); \
      return {};                                                        \
    }                                                                   \
  } while (false)

struct ConvOpts {
  int64_t num_dims = 3;
  int64_t input_dims[5];
  int64_t filter_dims[5];
  int64_t output_dims[5];
  int64_t bias_dims[5];
  int64_t input_strides[5];
  int64_t filter_strides[5];
  int64_t output_strides[5];
  int64_t bias_strides[5];
  int64_t paddings[3];
  int64_t strides[3];
  int64_t dilations[3];
  int64_t data_type;
  int64_t data_format;
  int64_t conv_kind;
  int64_t act_kind;

  int64_t input_size() {
    return std::accumulate(input_dims, input_dims + num_dims + 2, 1,
                           std::multiplies<int64_t>());
  }
  int64_t output_size() {
    return std::accumulate(output_dims, output_dims + num_dims + 2, 1,
                           std::multiplies<int64_t>());
  }
  int64_t filter_size() {
    return std::accumulate(filter_dims, filter_dims + num_dims + 2, 1,
                           std::multiplies<int64_t>());
  }
  int64_t bias_size() {
    return std::accumulate(bias_dims, bias_dims + num_dims + 2, 1,
                           std::multiplies<int64_t>());
  }
};

struct MatMulOpts {
  int64_t num_dims = 3;
  int64_t input0_dims[5];
  int64_t input1_dims[5];
  int64_t output_dims[5];
  int64_t bias_dims[5];
  int64_t input0_strides[5];
  int64_t input1_strides[5];
  int64_t output_strides[5];
  int64_t bias_strides[5];
  int64_t data_type;
  int64_t act_kind;

  int64_t input0_size() {
    return std::accumulate(input0_dims, input0_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t input1_size() {
    return std::accumulate(input1_dims, input1_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t output_size() {
    return std::accumulate(output_dims, output_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t bias_size() {
    return std::accumulate(bias_dims, bias_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
};
