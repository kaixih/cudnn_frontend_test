#include "util.h"

#include <cassert>

#include "cmd_options.h"

namespace {
std::vector<int64_t> GetIntegersFromString(const std::string& str) {
  std::vector<int64_t> results;
  std::stringstream ss(str);
  for (int i; ss >> i;) {
    results.push_back(i);
    if (ss.peek() == ',') ss.ignore();
  }
  return results;
}

std::vector<int64_t> ComputeOutputDims(std::vector<int64_t>& input_vec,
                                       std::vector<int64_t>& filter_vec,
                                       std::vector<int64_t>& stride_vec,
                                       std::vector<int64_t>& padding_vec,
                                       std::vector<int64_t>& dilation_vec) {
  assert(input_vec.size() == filter_vec.size());
  int tensor_rank = input_vec.size();
  std::vector<int64_t> results(tensor_rank);
  results[0] = input_vec[0];
  results[1] = filter_vec[0];
  for (int i = 0; i < tensor_rank - 2; i++) {
    results[i + 2] = (input_vec[i + 2] + (2 * padding_vec[i]) -
                      ((filter_vec[i + 2] - 1) * dilation_vec[i] + 1)) /
                         stride_vec[i] +
                     1;
  }
  return results;
}

std::vector<int64_t> ComputeOutputDims(std::vector<int64_t>& a_vec,
                                       std::vector<int64_t>& b_vec) {
  assert(a_vec.size() == b_vec.size());
  assert(a_vec.size() == 3);
  int tensor_rank = a_vec.size();
  std::vector<int64_t> c_vec(tensor_rank);
  c_vec[0] = a_vec[0];
  c_vec[1] = a_vec[1];
  c_vec[2] = b_vec[2];
  return c_vec;
}

std::vector<int64_t> ComputeOutputDims(std::vector<int64_t>& input_vec,
                                       std::vector<int64_t>& stride_vec,
                                       std::vector<int64_t>& padding_vec,
                                       std::vector<int64_t>& window_vec) {
  int tensor_rank = input_vec.size();
  std::vector<int64_t> results(tensor_rank);
  results[0] = input_vec[0];
  results[1] = input_vec[1];
  for (int i = 0; i < tensor_rank - 2; i++) {
    results[i + 2] = (input_vec[i + 2] + 2 * padding_vec[i] - window_vec[i]) /
                         stride_vec[i] + 1;
  }
  return results;
}

std::vector<int64_t> ComputeStrides(std::vector<int64_t>& vec,
                                    int64_t transpose = 0) {
  int tensor_dims = vec.size();
  std::vector<int64_t> strides(tensor_dims);

  if (transpose == 0) {
    strides[tensor_dims - 1] = 1;
    for (int64_t d = tensor_dims - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * vec[d + 1];
    }
  } else {
    strides[1] = 1;
    // For transposed access, vec[1] becomes the leading dim.
    strides[tensor_dims - 1] = strides[1] * vec[1];
    strides[0] = strides[2] * vec[2];
  }
  return strides;
}
}  // namespace

std::vector<int64_t> ComputeStrides(std::vector<int64_t>& vec,
                                    int data_format) {
  int tensor_dims = vec.size();
  std::vector<int64_t> strides(tensor_dims);
  if (data_format == 0) {
    // Channels first format: NCHW or NCDHW.
    strides[tensor_dims - 1] = 1;
    for (int64_t d = tensor_dims - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * vec[d + 1];
    }
  } else {
    // Channels last format: NHWC or NDHWC.
    strides[1] = 1;
    strides[tensor_dims - 1] = strides[1] * vec[1];
    for (int64_t d = tensor_dims - 2; d >= 2; d--) {
      strides[d] = strides[d + 1] * vec[d + 1];
    }
    strides[0] = strides[2] * vec[2];
  }
  return strides;
}

bool IsPrintAll() {
  static bool result = [] {
    bool print_on = false;
    const char* env_p = std::getenv("PRINTALL");
    if (env_p && (strcmp(env_p, "1") == 0 || strcmp(env_p, "true") == 0)) {
      print_on = true;
    }
    return print_on;
  }();
  return result;
}

ConvOpts ParseConvOpts(int argc, char** argv) {
  struct CmdConvOpts {
    std::string input_dims = "3,8,10,10";
    std::string filter_dims = "8,8,2,2";
    std::string bias_dims = "1,8,1,1";
    std::string paddings = "1,1";
    std::string strides = "1,1";
    std::string dilations = "1,1";
    int data_format = 1;
    int data_type = 1;
    int graph_index = 0;
    int engine_index = 0;
  };

  auto parser = CmdOpts<CmdConvOpts>::Create(
      {{"-input", &CmdConvOpts::input_dims},
       {"-filter", &CmdConvOpts::filter_dims},
       {"-bias", &CmdConvOpts::bias_dims},
       {"-padding", &CmdConvOpts::paddings},
       {"-stride", &CmdConvOpts::strides},
       {"-dilation", &CmdConvOpts::dilations},
       {"-data_format", &CmdConvOpts::data_format},
       {"-data_type", &CmdConvOpts::data_type},
       {"-graph_index", &CmdConvOpts::graph_index},
       {"-engine_index", &CmdConvOpts::engine_index}});

  auto parsed_opts = parser->parse(argc, argv);

  ConvOpts opts;
  // Get the dims from the parsed string.
  auto input_vec = GetIntegersFromString(parsed_opts.input_dims);
  std::copy(input_vec.begin(), input_vec.end(), opts.input_dims);
  auto filter_vec = GetIntegersFromString(parsed_opts.filter_dims);
  std::copy(filter_vec.begin(), filter_vec.end(), opts.filter_dims);
  auto bias_vec = GetIntegersFromString(parsed_opts.bias_dims);
  std::copy(bias_vec.begin(), bias_vec.end(), opts.bias_dims);
  auto stride_vec = GetIntegersFromString(parsed_opts.strides);
  std::copy(stride_vec.begin(), stride_vec.end(), opts.strides);
  auto padding_vec = GetIntegersFromString(parsed_opts.paddings);
  std::copy(padding_vec.begin(), padding_vec.end(), opts.paddings);
  auto dilation_vec = GetIntegersFromString(parsed_opts.dilations);
  std::copy(dilation_vec.begin(), dilation_vec.end(), opts.dilations);
  auto output_vec = ComputeOutputDims(input_vec, filter_vec, stride_vec,
                                      padding_vec, dilation_vec);
  std::copy(output_vec.begin(), output_vec.end(), opts.output_dims);

  // Compute the strides from the dims and format.
  auto i_stride_vec = ComputeStrides(input_vec, parsed_opts.data_format);
  std::copy(i_stride_vec.begin(), i_stride_vec.end(), opts.input_strides);
  auto f_stride_vec = ComputeStrides(filter_vec, parsed_opts.data_format);
  std::copy(f_stride_vec.begin(), f_stride_vec.end(), opts.filter_strides);
  auto o_stride_vec = ComputeStrides(output_vec, parsed_opts.data_format);
  std::copy(o_stride_vec.begin(), o_stride_vec.end(), opts.output_strides);
  auto b_stride_vec = ComputeStrides(bias_vec, parsed_opts.data_format);
  std::copy(b_stride_vec.begin(), b_stride_vec.end(), opts.bias_strides);

  opts.num_dims = input_vec.size() - 2;
  opts.data_type = parsed_opts.data_type;
  opts.data_format = parsed_opts.data_format;
  opts.graph_index = parsed_opts.graph_index;
  opts.engine_index = parsed_opts.engine_index;

  return opts;
}

void PrintConvOpts(ConvOpts& opts) {
  printf(">>> Retrieved CONVOLUTION specs:\n");
  auto print_ints = [](const int64_t* a, int n, const std::string& name) {
    printf(">>>   %s: ", name.c_str());
    for (int i = 0; i < n; i++) {
      printf("%ld, ", a[i]);
    }
    printf("\n");
  };
  print_ints(&opts.num_dims, 1, "num_spatial_dims");
  print_ints(opts.input_dims, opts.num_dims + 2, "input_dims (-input)");
  print_ints(opts.filter_dims, opts.num_dims + 2, "filter_dims (-filter)");
  // Only use the bias when graph has fused convolution.
  if (opts.graph_index > 2) {
    print_ints(opts.bias_dims, opts.num_dims + 2, "bias_dims (-bias)");
  }
  print_ints(opts.output_dims, opts.num_dims + 2, "output_dims (-output)");
  print_ints(opts.input_strides, opts.num_dims + 2, "input_strides");
  print_ints(opts.filter_strides, opts.num_dims + 2, "filter_strides");
  if (opts.graph_index > 2) {
    print_ints(opts.bias_strides, opts.num_dims + 2, "bias_strides");
  }
  print_ints(opts.output_strides, opts.num_dims + 2, "output_strides");
  print_ints(opts.paddings, opts.num_dims, "paddings (-padding)");
  print_ints(opts.strides, opts.num_dims, "strides (-stride)");
  print_ints(opts.dilations, opts.num_dims, "dilations (-dilation)");
  print_ints(&opts.data_type, 1, "data_type (-data_type [0<fp32>|1<fp16>])");
  print_ints(&opts.data_format, 1,
             "data_format (-data_format [0<nchw>|1<nhwc>])");
  print_ints(&opts.engine_index, 1, "engine_index (-engine_index)");
}

MatMulOpts ParseMatMulOpts(int argc, char** argv) {
  struct CmdMatMulOpts {
    std::string a_dims = "1,8,16";
    std::string b_dims = "1,16,32";
    std::string bias_dims = "1,1,32";
    bool transpose_a = 0;
    bool transpose_b = 0;
    int data_type = 1;
    int graph_index = 100;
    int engine_index = 0;
  };

  auto parser = CmdOpts<CmdMatMulOpts>::Create(
      {{"-a", &CmdMatMulOpts::a_dims},
       {"-b", &CmdMatMulOpts::b_dims},
       {"-bias", &CmdMatMulOpts::bias_dims},
       {"-transpose_a", &CmdMatMulOpts::transpose_a},
       {"-transpose_b", &CmdMatMulOpts::transpose_b},
       {"-data_type", &CmdMatMulOpts::data_type},
       {"-graph_index", &CmdMatMulOpts::graph_index},
       {"-engine_index", &CmdMatMulOpts::engine_index}});

  auto parsed_opts = parser->parse(argc, argv);

  MatMulOpts opts;
  // Get the dims from the parsed string.
  auto a_vec = GetIntegersFromString(parsed_opts.a_dims);
  std::copy(a_vec.begin(), a_vec.end(), opts.a_dims);
  auto b_vec = GetIntegersFromString(parsed_opts.b_dims);
  std::copy(b_vec.begin(), b_vec.end(), opts.b_dims);
  auto bias_vec = GetIntegersFromString(parsed_opts.bias_dims);
  std::copy(bias_vec.begin(), bias_vec.end(), opts.bias_dims);
  auto c_vec = ComputeOutputDims(a_vec, b_vec);
  std::copy(c_vec.begin(), c_vec.end(), opts.c_dims);

  opts.num_dims = a_vec.size();

  auto a_stride_vec = ComputeStrides(a_vec, parsed_opts.transpose_a);
  std::copy(a_stride_vec.begin(), a_stride_vec.end(), opts.a_strides);
  auto b_stride_vec = ComputeStrides(b_vec, parsed_opts.transpose_b);
  std::copy(b_stride_vec.begin(), b_stride_vec.end(), opts.b_strides);
  auto c_stride_vec = ComputeStrides(c_vec);
  std::copy(c_stride_vec.begin(), c_stride_vec.end(), opts.c_strides);
  auto bias_stride_vec = ComputeStrides(bias_vec);
  std::copy(bias_stride_vec.begin(), bias_stride_vec.end(), opts.bias_strides);

  opts.transpose_a = parsed_opts.transpose_a;
  opts.transpose_b = parsed_opts.transpose_b;
  opts.data_type = parsed_opts.data_type;
  opts.graph_index = parsed_opts.graph_index;
  opts.engine_index = parsed_opts.engine_index;

  return opts;
}

void PrintMatMulOpts(MatMulOpts& opts) {
  printf(">>> Retrieved MatMul specs:\n");
  auto print_ints = [](const int64_t* a, int n, const std::string& name) {
    printf(">>>   %s: ", name.c_str());
    for (int i = 0; i < n; i++) {
      printf("%ld, ", a[i]);
    }
    printf("\n");
  };
  print_ints(&opts.num_dims, 1, "num_dims");
  print_ints(opts.a_dims, opts.num_dims, "a_dims (-a)");
  print_ints(opts.b_dims, opts.num_dims, "b_dims (-b)");
  print_ints(opts.bias_dims, opts.num_dims, "bias_dims (-bias)");
  print_ints(opts.c_dims, opts.num_dims, "c_dims (-c)");
  print_ints(opts.a_strides, opts.num_dims, "a_strides");
  print_ints(opts.b_strides, opts.num_dims, "b_strides");
  print_ints(opts.bias_strides, opts.num_dims, "bias_strides");
  print_ints(opts.c_strides, opts.num_dims, "c_strides");
  print_ints(&opts.transpose_a, 1, "transpose_a (-transpose_a)");
  print_ints(&opts.transpose_b, 1, "transpose_b (-transpose_b)");
  print_ints(&opts.data_type, 1, "data_type (-data_type [0<fp32>|1<fp16>])");
  print_ints(&opts.engine_index, 1, "engine_index (-engine_index)");
}

std::string CudnnBehaviorNoteToString(cudnnBackendBehaviorNote_t status) {
  switch (status) {
    case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
      return "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
      return "CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER";
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
      return "CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER";
    default:
      return "<unknown cudnn behavior note>";
  }
}

ResampleOpts ParseResampleOpts(int argc, char** argv) {
  struct CmdResampleOpts {
    std::string input_dims = "3,8,10,10";
    std::string paddings = "1,1";
    std::string strides = "2,2";
    std::string window_sizes = "3,3";
    int data_format = 1;
    int data_type = 1;
    int graph_index = 200;
    int engine_index = 0;
  };

  auto parser = CmdOpts<CmdResampleOpts>::Create(
      {{"-input", &CmdResampleOpts::input_dims},
       {"-padding", &CmdResampleOpts::paddings},
       {"-stride", &CmdResampleOpts::strides},
       {"-window", &CmdResampleOpts::window_sizes},
       {"-data_format", &CmdResampleOpts::data_format},
       {"-data_type", &CmdResampleOpts::data_type},
       {"-graph_index", &CmdResampleOpts::graph_index},
       {"-engine_index", &CmdResampleOpts::engine_index}});

  auto parsed_opts = parser->parse(argc, argv);

  ResampleOpts opts;
  // Get the dims from the parsed string.
  auto input_vec = GetIntegersFromString(parsed_opts.input_dims);
  std::copy(input_vec.begin(), input_vec.end(), opts.input_dims);
  auto stride_vec = GetIntegersFromString(parsed_opts.strides);
  std::copy(stride_vec.begin(), stride_vec.end(), opts.strides);
  auto padding_vec = GetIntegersFromString(parsed_opts.paddings);
  std::copy(padding_vec.begin(), padding_vec.end(), opts.paddings);
  auto window_vec = GetIntegersFromString(parsed_opts.window_sizes);
  std::copy(window_vec.begin(), window_vec.end(), opts.window_sizes);
  auto output_vec = ComputeOutputDims(input_vec, stride_vec, padding_vec,
                                      window_vec);
  std::copy(output_vec.begin(), output_vec.end(), opts.output_dims);

  // Compute the strides from the dims and format.
  auto i_stride_vec = ComputeStrides(input_vec, parsed_opts.data_format);
  std::copy(i_stride_vec.begin(), i_stride_vec.end(), opts.input_strides);
  auto o_stride_vec = ComputeStrides(output_vec, parsed_opts.data_format);
  std::copy(o_stride_vec.begin(), o_stride_vec.end(), opts.output_strides);

  opts.num_dims = input_vec.size() - 2;
  opts.data_type = parsed_opts.data_type;
  opts.data_format = parsed_opts.data_format;
  opts.graph_index = parsed_opts.graph_index;
  opts.engine_index = parsed_opts.engine_index;

  return opts;
}

void PrintResampleOpts(ResampleOpts& opts) {
  printf(">>> Retrieved RESAMPLE specs:\n");
  auto print_ints = [](const int64_t* a, int n, const std::string& name) {
    printf(">>>   %s: ", name.c_str());
    for (int i = 0; i < n; i++) {
      printf("%ld, ", a[i]);
    }
    printf("\n");
  };
  print_ints(&opts.num_dims, 1, "num_spatial_dims");
  print_ints(opts.input_dims, opts.num_dims + 2, "input_dims (-input)");
  print_ints(opts.output_dims, opts.num_dims + 2, "output_dims (-output)");
  print_ints(opts.input_strides, opts.num_dims + 2, "input_strides");
  print_ints(opts.output_strides, opts.num_dims + 2, "output_strides");
  print_ints(opts.paddings, opts.num_dims, "paddings (-padding)");
  print_ints(opts.strides, opts.num_dims, "strides (-stride)");
  print_ints(opts.window_sizes, opts.num_dims, "window_sizes (-window)");
  print_ints(&opts.data_type, 1, "data_type (-data_type [0<fp32>|1<fp16>])");
  print_ints(&opts.data_format, 1,
             "data_format (-data_format [0<nchw>|1<nhwc>])");
  print_ints(&opts.engine_index, 1, "engine_index (-engine_index)");
}

std::string CudnnNumericalNoteToString(cudnnBackendNumericalNote_t status) {
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
      return "<unknown cudnn numerical note>";
  }
}

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

cudnnDataType_t ToCudnnDataType(int data_type) {
  if (data_type == 0) {
    return CUDNN_DATA_FLOAT;
  } else if (data_type == 2) {
    return CUDNN_DATA_INT8;
  }

  return CUDNN_DATA_HALF;
}

cudnnDataType_t GetConvAccumulatorCudnnDataType(int data_type) {
  // We always use fp32 as the accumulator dtype for both fp16 and fp32 inputs.
  return CUDNN_DATA_FLOAT;
}

cudnnDataType_t GetConvActivationCudnnDataType(int data_type) {
  // We always use fp32 as the activation dtype for both fp16 and fp32 inputs.
  return CUDNN_DATA_FLOAT;
}

float InitOnes(int i) { return 1.f; }
float InitZeros(int i) { return 0.f; }
float InitRandoms(int i) { return static_cast<float>(rand()) / RAND_MAX; }
float InitSeq(int i) { return static_cast<float>(i); }
float InitConstant(int i) { return 0.001f; }
