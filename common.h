#include <cudnn.h>
#include <functional>

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
      std::cerr << "Error on line " << __LINE__ << ": "      \
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
    if (!(_status == CUDNN_STATUS_SUCCESS)) {            \
      std::ostringstream oss;                                           \
      oss << CudnnStatusToString(_status) << "\nin " << __FILE__ << "(" \
          << __LINE__ << "): '" << #expr << "' " << (expr).get_error(); \
      return {};             \
    }                                                                   \
  } while (false)

struct ConvOpts {
  int64_t num_dims = 3;
  int64_t input_dims[5];
  int64_t filter_dims[5];
  int64_t output_dims[5];
  int64_t input_strides[5];
  int64_t filter_strides[5];
  int64_t output_strides[5];
  int64_t paddings[3];
  int64_t strides[3];
  int64_t dilations[3];
  int64_t data_type;
  int64_t data_format;
  int64_t conv_kind;

  int64_t input_size() {
    int64_t total = input_dims[0];
    for (int i = 1; i < num_dims + 2; i++) total *= input_dims[i];
    return total;
  }
  int64_t output_size() {
    int64_t total = output_dims[0];
    for (int i = 1; i < num_dims + 2; i++) total *= output_dims[i];
    return total;
  }
  int64_t filter_size() {
    int64_t total = filter_dims[0];
    for (int i = 1; i < num_dims + 2; i++) total *= filter_dims[i];
    return total;
  }
};

std::string CudnnStatusToString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
    default:
      return "<unknown cudnn status>";
  }
}

std::string CudnnStatusToString(cudnnBackendNumericalNote_t status) {
  switch (status) {
    case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
      return "CUDNN_NUMERICAL_NOTE_TENSOR_CORE";
    case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
      return "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS";
    case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
      return "CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION";
    case CUDNN_NUMERICAL_NOTE_FFT:
      return "CUDNN_NUMERICAL_NOTE_FFT";
    case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
      return "CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD:
      return "CUDNN_NUMERICAL_NOTE_WINOGRAD";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
      return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
      return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6";
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
      return "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13";
    default:
      return "<unknown cudnn status>";
  }
}

std::string CudnnStatusToString(cudnnBackendBehaviorNote_t status) {
  switch (status) {
    case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
      return "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
      return "CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
      return "CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER";
    default:
      return "<unknown cudnn status>";
  }
}

cudnnBackendDescriptorType_t GetCudnnConvolutionType(int64_t kind) {
  cudnnBackendDescriptorType_t conv_kind;
  switch (kind) {
    case 0: {
      conv_kind = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
      break;
    }
    case 1: {
      conv_kind =
          CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
      break;
    }
    case 2: {
      conv_kind = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
      break;
    }
  }
  return conv_kind;
}

cudnnDataType_t ToCudnnDataType(int data_type) {
  cudnnDataType_t dtype;
  switch (data_type) {
    case 0:
      dtype = CUDNN_DATA_FLOAT;
    case 1:
      dtype = CUDNN_DATA_HALF;
  }
  return dtype;
}

int GetConvAccumulatorType(int data_type) {
  int dtype;
  switch (data_type) {
    case 0:
      dtype = 0;
    case 1:
      dtype = 0;
  }
  return dtype;
}

template <typename T>
void InitDeviceTensor(void** d_ptr, size_t n, std::function<float()> init_fn) {
  checkCUDA(cudaMalloc(d_ptr, n * sizeof(T)));
  T* h_ptr = new T[n];
  for (size_t i = 0; i < n; i++) {
    h_ptr[i] = init_fn();
  }
  checkCUDA(cudaMemcpy(*d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice));
  delete[] h_ptr;
}

// Some predefined initialization functions.
float InitOnes() { return 1.f; }
float InitRandoms() { return static_cast<float>(rand()) / RAND_MAX; }

template <typename T>
void PrintDeviceTensor(void* d_ptr, size_t n, const std::string& prompt) {
  T* h_ptr = new T[n];
  checkCUDA(cudaMemcpy(h_ptr, d_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost));
  std::cout << prompt << std::endl;
  for (int i = 0; i < n; i++) {
    std::cout << static_cast<float>(h_ptr[i]) << " ";
    if ((i + 1) % 10 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  delete[] h_ptr;
}

