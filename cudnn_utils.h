#include <cudnn.h>

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
      break;
    case 1:
      dtype = CUDNN_DATA_HALF;
      break;
  }
  return dtype;
}

int GetConvAccumulatorType(int data_type) {
  int dtype;
  switch (data_type) {
    case 0:
      dtype = 0;
      break;
    case 1:
      dtype = 0;
      break;
  }
  return dtype;
}

int GetConvActivationType(int data_type) {
  int dtype;
  switch (data_type) {
    case 0:
      dtype = 0;
      break;
    case 1:
      dtype = 0;
      break;
  }
  return dtype;
}
