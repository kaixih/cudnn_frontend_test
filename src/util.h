#ifndef CUDNN_FE_TEST_SRC_UTIL_H_
#define CUDNN_FE_TEST_SRC_UTIL_H_

#include <cuda_fp16.h>
#include <cudnn.h>

#include <functional>
#include <iostream>
#include <numeric>

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

#define checkCUDA(expression)                                        \
  {                                                                  \
    cudaError_t status = (expression);                               \
    if (status != cudaSuccess) {                                     \
      std::cerr << RED "Error on line " << __LINE__ << ": "          \
                << cudaGetErrorString(status) << RESET << std::endl; \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  }

#define checkCUDNN(expression)                                                \
  {                                                                           \
    cudnnStatus_t status = (expression);                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                                     \
      std::cerr << RED "Error from file: " << __FILE__ << " on "              \
                << "line " << __LINE__ << ": " << cudnnGetErrorString(status) \
                << RESET << std::endl;                                        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  }

#define UNIQUE_VAR_NAME(x, y) UNIQUE_VAR_NAME_IMPL(x, y)
#define UNIQUE_VAR_NAME_IMPL(x, y) x##y

#define ASSIGN_OR_RETURN_IMPL(lhs, val_or, rexpr, err)    \
  auto val_or = (rexpr);                                  \
  if (!val_or.has_value()) {                              \
    std::cerr << RED "!!! " << err << RESET << std::endl; \
  }                                                       \
  lhs = std::move(val_or.value());

#define ASSIGN_OR_RETURN(lhs, rexpr, err) \
  ASSIGN_OR_RETURN_IMPL(lhs, UNIQUE_VAR_NAME(_val_or, __COUNTER__), rexpr, err)

#define RETURN_MSG_IF_CUDNN_ERROR(expr)                                       \
  do {                                                                        \
    cudnnStatus_t _status = (expr).get_status();                              \
    if (!(_status == CUDNN_STATUS_SUCCESS)) {                                 \
      std::ostringstream oss;                                                 \
      std::cerr << RED << CudnnStatusToString(_status) << "\nin " << __FILE__ \
                << "(" << __LINE__ << "): '" << #expr << "' "                 \
                << (expr).get_error() << RESET << std::endl;                  \
      return {};                                                              \
    }                                                                         \
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
  int64_t graph_index;
  int64_t engine_index;

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
  int64_t a_dims[5];
  int64_t b_dims[5];
  int64_t c_dims[5];
  int64_t bias_dims[5];
  int64_t a_strides[5];
  int64_t b_strides[5];
  int64_t c_strides[5];
  int64_t bias_strides[5];
  int64_t data_type;
  int64_t transpose_a;
  int64_t transpose_b;
  int64_t graph_index;
  int64_t engine_index;

  int64_t a_size() {
    return std::accumulate(a_dims, a_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t b_size() {
    return std::accumulate(b_dims, b_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t c_size() {
    return std::accumulate(c_dims, c_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
  int64_t bias_size() {
    return std::accumulate(bias_dims, bias_dims + num_dims, 1,
                           std::multiplies<int64_t>());
  }
};

bool IsPrintAll();

ConvOpts ParseConvOpts(int argc, char** argv);
void PrintConvOpts(ConvOpts& opts);

MatMulOpts ParseMatMulOpts(int argc, char** argv);
void PrintMatMulOpts(MatMulOpts& opts);

std::string CudnnBehaviorNoteToString(cudnnBackendBehaviorNote_t status);
std::string CudnnNumericalNoteToString(cudnnBackendNumericalNote_t status);
std::string CudnnStatusToString(cudnnStatus_t status);

cudnnDataType_t ToCudnnDataType(int data_type);
cudnnDataType_t GetConvAccumulatorCudnnDataType(int data_type);
cudnnDataType_t GetConvActivationCudnnDataType(int data_type);

float InitOnes(int i);
float InitZeros(int i);
float InitRandoms(int i);
float InitSeq(int i);

template <typename T>
void AllocTensorWithInitValues(void** d_ptr, size_t n,
                               std::function<float(int)> init_fn) {
  checkCUDA(cudaMalloc(d_ptr, n * sizeof(T)));
  T* h_ptr = new T[n];
  for (size_t i = 0; i < n; i++) {
    h_ptr[i] = static_cast<T>(init_fn(i));
  }
  checkCUDA(cudaMemcpy(*d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice));
  delete[] h_ptr;
}

template <typename T>
void PrintTensor(void* d_ptr, size_t n, const std::string& prompt) {
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

template void AllocTensorWithInitValues<float>(
    void** d_ptr, size_t n, std::function<float(int)> init_fn);
template void AllocTensorWithInitValues<__half>(
    void** d_ptr, size_t n, std::function<float(int)> init_fn);
template void PrintTensor<float>(void* d_ptr, size_t n,
                                 const std::string& prompt);
template void PrintTensor<__half>(void* d_ptr, size_t n,
                                  const std::string& prompt);

#endif  // CUDNN_FE_TEST_SRC_UTIL_H_
