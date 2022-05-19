#include <iostream>
#include <cudnn_frontend.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <map>

static int checkCudaError(cudaError_t code, const char* expr, const char* file,
                          int line) {
  if (code) {
    printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code,
           cudaGetErrorString(code), expr);
    return 1;
  }
  return 0;
}

static int checkCudnnError(cudnnStatus_t code, const char* expr,
                           const char* file, int line) {
  if (code) {
    printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line,
           (int)code, cudnnGetErrorString(code), expr);
    return 1;
  }
  return 0;
}

std::string ToString(cudnnStatus_t status) {
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

#define checkCudaErr(...)                                                    \
  do {                                                                       \
    int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    if (err) {                                                               \
      numErrors++;                                                           \
    }                                                                        \
  } while (0)

#define checkCudnnErr(...)                                                    \
  do {                                                                        \
    int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    if (err) {                                                                \
      numErrors++;                                                            \
    }                                                                         \
  } while (0)

#define RETURN_IF_CUDNN_ERROR(expr) \
  do { \
    cudnnStatus_t status = expr; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      std::cout << "Error: " << ToString(status) << " ln:" << __LINE__ << std::endl; \
      exit(1); \
    } \
  } while (false)


static inline int getFwdConvPaddedImageDim(int tensorDim, int pad) {
  return tensorDim + (2 * pad);
}

static inline int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
  return ((filterDim - 1) * dilation) + 1;
}

static inline int getFwdConvOutputDim(
    int tensorDim, 
    int pad, 
    int filterDim, 
    int stride, 
    int dilation) {
  int p = (getFwdConvPaddedImageDim(tensorDim, pad) -
           getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
  return (p);
}

static void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims,
                            cudnnTensorFormat_t filterFormat) {
  if (filterFormat == CUDNN_TENSOR_NCHW ||
      filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
    strideA[nbDims - 1] = 1;
    for (int64_t d = nbDims - 2; d >= 0; d--) {
      strideA[d] = strideA[d + 1] * dimA[d + 1];
    }
  } else {
    strideA[1]          = 1;
    strideA[nbDims - 1] = strideA[1] * dimA[1];
    for (int64_t d = nbDims - 2; d >= 2; d--) {
      strideA[d] = strideA[d + 1] * dimA[d + 1];
    }
    strideA[0] = strideA[2] * dimA[2];
  }
}

template<typename T>
void init_me(T *hostI, int insize, T *hostF, int filtersize,
             T *hostO, int outsize) {
  for (int i = 0; i < insize; i++) {
    hostI[i] = 1;
  }
  for (int i = 0; i < filtersize; i++) {
    hostF[i] = 1;
  }
  for (int i = 0; i < outsize; i++) {
    hostO[i] = 1;
  }
}

template<>
void init_me<__half>(__half *hostI, int insize, __half *hostF, int filtersize,
                     __half *hostO, int outsize) {
  for (int i = 0; i < insize; i++) {
    hostI[i] = __float2half(1);
  }
  for (int i = 0; i < filtersize; i++) {
    hostF[i] = __float2half(1);
  }
  for (int i = 0; i < outsize; i++) {
    hostO[i] = __float2half(1);
  }
}

template<typename T>
void print_me(T *o_h, int o_size) {
  std::cout << "Output (float/double): ";
  for (int i = 0; i < std::min(o_size, 30); i++) {
    std::cout << o_h[i] << " ";
  }
  std::cout << std::endl;
}

template<>
void print_me<__half>(__half *o_h, int o_size) {
  std::cout << "Output (half): ";
  for (int i = 0; i < std::min(o_size, 30); i++) {
    std::cout << __half2float(o_h[i]) << " ";
  }
  std::cout << std::endl;
}

template<typename DTYPE, bool use_nhwc=false,
         bool use_float=std::is_floating_point<DTYPE>::value>
void run_me(int engine_id) {
  int mathType  = 0;

  int64_t dimA[] = {8, 64, 128, 128, 128};
  int64_t padA[]        = {1, 1, 1};
  int64_t convstrideA[] = {1, 1, 1};
  int64_t filterdimA[] = {32, 64, 3, 3, 3};

  cudnnTensorFormat_t filterFormat = use_nhwc ? CUDNN_TENSOR_NHWC :
                                                CUDNN_TENSOR_NCHW;

  cudnnConvolutionMode_t mode      = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t dataType;
  cudnnDataType_t convType;
  if (use_float && sizeof(DTYPE) == 8) {
    dataType = CUDNN_DATA_DOUBLE;
    convType = CUDNN_DATA_DOUBLE;
  } else if (use_float && sizeof(DTYPE) == 4) {
    dataType = CUDNN_DATA_FLOAT;
    convType = CUDNN_DATA_FLOAT;
  } else {
    dataType = CUDNN_DATA_HALF;
    convType = CUDNN_DATA_FLOAT;
  }

  std::map<int, std::string> note_map;
  note_map[0] = "CUDNN_NUMERICAL_NOTE_TENSOR_CORE";
  note_map[1] = "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS";
  note_map[2] = "CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION";
  note_map[3] = "CUDNN_NUMERICAL_NOTE_FFT";
  note_map[4] = "CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC";
  note_map[5] = "CUDNN_NUMERICAL_NOTE_WINOGRAD";
  note_map[6] = "CUDNN_NUMERICAL_NOTE_TYPE_COUNT";

  cudnnHandle_t handle_ = nullptr;
  DTYPE* devPtrI;
  DTYPE* devPtrF;
  DTYPE* devPtrO;
  DTYPE* hostI;
  DTYPE* hostF;
  DTYPE* hostO;

  constexpr int convDim = 3;
  float alpha     = 1.0f;
  float beta      = 0.0;
  int numErrors   = 0;
  int64_t dilationA[] = {1, 1, 1};
  int insize      = 0;
  int filtersize  = 0;
  int64_t outdimA[]   = {-1, -1, -1, -1, -1};
  int outsize     = 0;

  int64_t dimA_padded[5];
  int64_t outdimA_padded[5];
  int64_t filterdimA_padded[5];
  int64_t strideA_padded[5];
  int64_t outstrideA_padded[5];
  int64_t filterstrideA_padded[5];

  auto conv_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
  // auto conv_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
  // auto conv_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;

  outdimA[0] = dimA[0];
  outdimA[1] = filterdimA[0];
  for (int dim = 0; dim < 3; dim++) {
    outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim],
                                           filterdimA[dim + 2],
                                           convstrideA[dim], dilationA[dim]);
  }
  std::cout << "Calculated output dims: ";
  for (int i = 0; i < 5; i++) {
    std::cout << outdimA[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 5; i++) {
    dimA_padded[i]       = dimA[i];
    outdimA_padded[i]    = outdimA[i];
    filterdimA_padded[i] = filterdimA[i];
  }

  generateStrides(dimA_padded, strideA_padded, 5, filterFormat);
  insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3] *
           dimA_padded[4];
  std::cout << "Calculated strideA_padded: ";
  for (int i = 0; i < 5; i++) {
    std::cout << strideA_padded[i] << " ";
  }
  std::cout << std::endl;


  generateStrides(filterdimA_padded, filterstrideA_padded, 5, filterFormat);
  filtersize = filterdimA_padded[0] * filterdimA_padded[1] *
               filterdimA_padded[2] * filterdimA_padded[3] *
               filterdimA_padded[4];
  std::cout << "Calculated filterstrideA_padded: ";
  for (int i = 0; i < 5; i++) {
    std::cout << filterstrideA_padded[i] << " ";
  }
  std::cout << std::endl;

  generateStrides(outdimA_padded, outstrideA_padded, 5, filterFormat);
  outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] *
            outdimA_padded[3] * outdimA_padded[4];
  std::cout << "Calculated outstrideA_padded: ";
  for (int i = 0; i < 5; i++) {
    std::cout << outstrideA_padded[i] << " ";
  }
  std::cout << std::endl;

  checkCudaErr(cudaMalloc((void**)&(devPtrI), (insize) * sizeof(devPtrI[0])));
  checkCudaErr(cudaMalloc((void**)&(devPtrF),
                          (filtersize) * sizeof(devPtrF[0])));
  checkCudaErr(cudaMalloc((void**)&(devPtrO), (outsize) * sizeof(devPtrO[0])));

  hostI = (DTYPE*)calloc(insize, sizeof(hostI[0]));
  hostF = (DTYPE*)calloc(filtersize, sizeof(hostF[0]));
  hostO = (DTYPE*)calloc(outsize, sizeof(hostO[0]));
  init_me<DTYPE>(hostI, insize, hostF, filtersize, hostO, outsize);

  checkCudaErr(cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize,
                          cudaMemcpyHostToDevice));
  checkCudaErr(cudaMemcpy(devPtrF, hostF, sizeof(hostF[0]) * filtersize,
                          cudaMemcpyHostToDevice));
  checkCudaErr(cudaMemcpy(devPtrO, hostO, sizeof(hostO[0]) * outsize,
                          cudaMemcpyHostToDevice));
  checkCudaErr(cudaDeviceSynchronize());

  checkCudnnErr(cudnnCreate(&handle_));
  auto tensor_x = cudnn_frontend::TensorBuilder()
                      .setDim(5, dimA_padded)
                      .setStrides(5, strideA_padded)
                      .setId('x')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  std::cout << tensor_x.get_error() << std::endl;
	RETURN_IF_CUDNN_ERROR(tensor_x.get_status());
  std::cout << tensor_x.describe() << std::endl;

  auto tensor_y = cudnn_frontend::TensorBuilder()
                      .setDim(5, outdimA_padded)
                      .setStrides(5, outstrideA_padded)
                      .setId('y')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
	RETURN_IF_CUDNN_ERROR(tensor_y.get_status());
  std::cout << tensor_y.describe() << std::endl;

  auto tensor_w = cudnn_frontend::TensorBuilder()
                      .setDim(5, filterdimA_padded)
                      .setStrides(5, filterstrideA_padded)
                      .setId('w')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
	RETURN_IF_CUDNN_ERROR(tensor_w.get_status());
  std::cout << tensor_w.describe() << std::endl;

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                      .setDataType(convType)
                      .setMathMode(mode)
                      .setNDims(convDim)
                      .setStrides(convDim, convstrideA)
                      .setPrePadding(convDim, padA)
                      .setPostPadding(convDim, padA)
                      .setDilation(convDim, dilationA)
                      .build();
	RETURN_IF_CUDNN_ERROR(conv_desc.get_status());
  std::cout << conv_desc.describe() << std::endl;

  // Build Operation
  auto op = cudnn_frontend::OperationBuilder(conv_mode)
      .setxDesc(tensor_x)
      .setyDesc(tensor_y)
      .setwDesc(tensor_w)
      .setcDesc(conv_desc)
      .setAlpha((DTYPE)alpha)
      .setBeta((DTYPE)beta)
      .build();
	RETURN_IF_CUDNN_ERROR(op.get_status());
  std::cout << op.describe() << std::endl;

  // Build OperationGraph
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                   .setHandle(handle_)
                   .setOperationGraph(ops.size(), ops.data())
                   .build();
	RETURN_IF_CUDNN_ERROR(opGraph.get_status());
  std::cout << opGraph.describe() << std::endl;

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(opGraph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();
	RETURN_IF_CUDNN_ERROR(heuristics.get_status());
  std::cout << "Heuristic has " << heuristics.getEngineConfigCount()
            << " configurations " << std::endl;

  auto ec_count = heuristics.getEngineConfigCount();
  auto &engine_config = heuristics.getEngineConfig(ec_count);
  std::cout << "Heuristic size: " << engine_config.size() << std::endl;

  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
      .setOperationGraph(opGraph)
      .setOperation(
          conv_mode)
      .build();
  auto &fallback_list = fallback.getFallbackList();
  std::cout << "fallback size: " << fallback_list.size() << std::endl;

  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_frontend::filter(engine_config, filtered_configs,
                         [](cudnnBackendDescriptor_t engine_config){ return false; });
  cudnn_frontend::filter(fallback_list, filtered_configs,
                         [](cudnnBackendDescriptor_t engine_config){ return false; });
  std::cout << "filter left: " << filtered_configs.size() << std::endl;

  std::vector<cudnn_frontend::ExecutionPlan> cached_plans;
  for (int i = 0; i < filtered_configs.size(); i++) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle_)
                    .setEngineConfig(filtered_configs[i], opGraph.getTag())
                    .build();
    std::cout << "Building Plan tag (" << plan.getTag() << "): ";
    cudnnStatus_t status = plan.get_status();
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cout << "Fail. ";
      continue;
    } else {
      std::cout << "Success. ";
    }
    cached_plans.push_back(std::move(plan));
    std::cout << "WorkspaceSize(bytes): " << cached_plans.back().getWorkspaceSize() << std::endl;
  }
  std::cout << "Cached plan size: " << cached_plans.size() << std::endl;

  auto plan_desc = cached_plans[engine_id].get_raw_desc();
  std::cout << "Do convolution with plan: " << cached_plans[engine_id].getTag();

  auto workspace_size = cached_plans[engine_id].getWorkspaceSize(); 
  std::cout << "\nWorkspace size: " << workspace_size << std::endl;

  char *ws_ptr = nullptr;
  if (workspace_size != 0) {
    checkCudaErr(cudaMalloc((void**)&(ws_ptr), workspace_size));
  }
  void * data_ptrs[] = {devPtrI, devPtrO, devPtrF};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
                         .setWorkspacePointer((void*)ws_ptr)
                         .setDataPointers(3, data_ptrs)
                         .setUids(3, uids)
                         .build();
	RETURN_IF_CUDNN_ERROR(variantPack.get_status());
  std::cout << "variantPack " << variantPack.describe() << std::endl;

  auto ret = cudnnBackendExecute(handle_, plan_desc,
                                    variantPack.get_raw_desc());
	RETURN_IF_CUDNN_ERROR(ret);

  DTYPE *o_d;
  DTYPE *o_h;
  int o_size;

  if (conv_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
    std::tie(o_d, o_h, o_size) = std::make_tuple(devPtrO, hostO, outsize);
  } else if (conv_mode ==
             CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
    std::tie(o_d, o_h, o_size) = std::make_tuple(devPtrF, hostF, filtersize);
  } else if (conv_mode ==
             CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
    std::tie(o_d, o_h, o_size) = std::make_tuple(devPtrI, hostI, insize);
  }

  checkCudaErr(cudaMemcpy(o_h, o_d, sizeof(o_h[0]) * o_size,
                          cudaMemcpyDeviceToHost));
  checkCudaErr(cudaDeviceSynchronize());
  print_me<DTYPE>(o_h, o_size);
  

  if (devPtrI) cudaFree(devPtrI);
  if (devPtrF) cudaFree(devPtrF);
  if (devPtrO) cudaFree(devPtrO);
  if (hostI) free(hostI);
  if (hostF) free(hostF);
  if (hostO) free(hostO);
  if (handle_) cudnnDestroy(handle_);
}

int main(int argc, char** argv) {
  int engine_id = 0;
  if (argc > 1) {
    engine_id = atoi(argv[1]);
  }
  run_me<float, /*use_nhwc*/true>(engine_id);
  /*
  cudnnTest -Rdgrad -algo0 -Pins -Pouts -Pcomps -Pmath1 -x -dim3 -dimA2,1,2,6,1 -strideA12,12,6,1,1 -filtA1,1,2,2,1 -filtFormat0 -padA0,0,0 -convStrideA1,1,1 -dilationA1,1,1 -A1 -B0 -strideOut5,5,5,1,1 -b
  */
}
