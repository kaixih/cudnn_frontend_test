#include <cuda_fp16.h>

#include <iostream>

#include "cmd_options.h"
#include "common.h"
#include "convolution_unfused.h"

void ComputeOutputDims(ConvOpts& opts) {
  opts.output_dims[0] = opts.input_dims[0];
  opts.output_dims[1] = opts.filter_dims[0];
  for (int i = 0; i < opts.num_dims; i++) {
    opts.output_dims[i + 2] =
        (opts.input_dims[i + 2] + (2 * opts.paddings[i]) -
         ((opts.filter_dims[i + 2] - 1) * opts.dilations[i] + 1)) /
            opts.strides[i] +
        1;
  }
}

void ComputeStrides(ConvOpts& opts, int data_format) {
  auto compute_strides = [&](const int64_t* dims, int64_t* strides) {
    int tensor_dims = opts.num_dims + 2;
    if (data_format == 0) {
      strides[tensor_dims - 1] = 1;
      for (int64_t d = tensor_dims - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
      }
    } else {
      strides[1] = 1;
      strides[tensor_dims - 1] = strides[1] * dims[1];
      for (int64_t d = tensor_dims - 2; d >= 2; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
      }
      strides[0] = strides[2] * dims[2];
    }
  };
  compute_strides(opts.input_dims, opts.input_strides);
  compute_strides(opts.filter_dims, opts.filter_strides);
  compute_strides(opts.output_dims, opts.output_strides);
}

std::optional<ConvOpts> ParseConvOpts(int argc, char** argv) {
  struct CmdConvOpts {
    std::string input_dims = "3,4,5,5";
    std::string filter_dims = "4,4,2,2,2";
    std::string paddings = "1,1,1";
    std::string strides = "1,1,1";
    std::string dilations = "1,1,1";
    int data_format = 0;
    int data_type = 0;
    int conv_kind = 0;
  };
  auto parser = CmdOpts<CmdConvOpts>::Create(
      {{"--input", &CmdConvOpts::input_dims},
       {"--filter", &CmdConvOpts::filter_dims},
       {"--stride", &CmdConvOpts::strides},
       {"--padding", &CmdConvOpts::paddings},
       {"--dilation", &CmdConvOpts::dilations},
       {"--data_format", &CmdConvOpts::data_format},
       {"--data_type", &CmdConvOpts::data_type},
       {"--conv_kind", &CmdConvOpts::conv_kind}});
  auto parsed_opts = parser->parse(argc, argv);
  if (!(parsed_opts.data_type >= 0 && parsed_opts.data_type <= 1)) {
    std::cout << "!!! --data_type: 0=float, 1=half, but we got "
              << parsed_opts.data_type << std::endl;
    return {};
  }
  if (!(parsed_opts.data_format >= 0 && parsed_opts.data_format <= 1)) {
    std::cout << "!!! --data_format: 0=nchw, 1=nhwc, but we got "
              << parsed_opts.data_type << std::endl;
    return {};
  }
  if (!(parsed_opts.conv_kind >= 0 && parsed_opts.conv_kind <= 2)) {
    std::cout << "!!! --conv_kind: 0=fwd, 1=bwd_filter, 2=bwd_input, "
              << "but we got " << parsed_opts.data_type << std::endl;
    return {};
  }

  ConvOpts opts;
  auto str2int_parser = [&](const std::string& str, int64_t* dst_array,
                            bool update_num_dims = false) {
    std::stringstream ss(str);
    int index = 0;
    for (int i; ss >> i;) {
      dst_array[index++] = i;
      if (ss.peek() == ',') ss.ignore();
    }
    if (update_num_dims) {
      opts.num_dims = index - 2;
    }
  };
  // We use "input_dims" to determine the convolution dim, i.e. 2D or 3D.
  str2int_parser(parsed_opts.input_dims, opts.input_dims, true);
  str2int_parser(parsed_opts.filter_dims, opts.filter_dims);
  str2int_parser(parsed_opts.strides, opts.strides);
  str2int_parser(parsed_opts.paddings, opts.paddings);
  str2int_parser(parsed_opts.dilations, opts.dilations);

  ComputeOutputDims(opts);
  ComputeStrides(opts, parsed_opts.data_format);
  opts.data_type = parsed_opts.data_type;
  opts.data_format = parsed_opts.data_format;
  opts.conv_kind = parsed_opts.conv_kind;

  return opts;
}

int ParseEngineOpts(int argc, char** argv) {
  struct CmdEngineOpts {
    int engine_index = 0;
  };
  auto parser = CmdOpts<CmdEngineOpts>::Create(
      {{"--engine_index", &CmdEngineOpts::engine_index}});
  auto parsed_opts = parser->parse(argc, argv);

  return parsed_opts.engine_index;
}

void PrintConvOpts(ConvOpts& opts) {
  std::cout << ">>> CONVOLUTION:" << std::endl;
  auto print_ints = [](const int64_t* a, int n, const std::string& name) {
    std::cout << ">>>   " << name << ": ";
    for (int i = 0; i < n; i++) {
      std::cout << a[i] << ", ";
    }
    std::cout << std::endl;
  };
  print_ints(&opts.num_dims, 1, "num_dims");
  print_ints(opts.input_dims, opts.num_dims + 2, "input_dims");
  print_ints(opts.filter_dims, opts.num_dims + 2, "filter_dims");
  print_ints(opts.output_dims, opts.num_dims + 2, "output_dims");
  print_ints(opts.input_strides, opts.num_dims + 2, "input_strides");
  print_ints(opts.filter_strides, opts.num_dims + 2, "filter_strides");
  print_ints(opts.output_strides, opts.num_dims + 2, "output_strides");
  print_ints(opts.paddings, opts.num_dims, "paddings");
  print_ints(opts.strides, opts.num_dims, "strides");
  print_ints(opts.dilations, opts.num_dims, "dilations");
  print_ints(&opts.data_type, 1, "data_type(0=float,1=half)");
  print_ints(&opts.data_format, 1, "data_format(0=nchw,1=nhwc)");
  print_ints(&opts.conv_kind, 1, "conv_kind(0=fwd,1=bwd_filter,2=bwd_input)");
}

int main(int argc, char** argv) {
  ASSIGN_OR_RETURN(auto opts, ParseConvOpts(argc, argv),
                   "Failed to parse the conv parameters.")
  bool print_on = false;
  const char* env_p = std::getenv("PRINTALL");
  if (env_p && (strcmp(env_p, "1") == 0 || strcmp(env_p, "true") == 0)) {
    print_on = true;
  }

  PrintConvOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));
  ASSIGN_OR_RETURN(auto op_graph, GetUnfusedConvGraph(opts, cudnn),
                   "Failed to build the conv graph.")
  std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>> plans;
  CreateOpRunners(cudnn, std::move(op_graph), &plans);

  int engine_index = ParseEngineOpts(argc, argv);
  std::cout << "Using (" << engine_index
            << "): " << plans[engine_index]->getTag() << std::endl;
  auto plan_desc = plans[engine_index]->get_raw_desc();
  auto workspace_size = plans[engine_index]->getWorkspaceSize();
  void* ws_ptr = nullptr;
  if (workspace_size != 0) {
    checkCUDA(cudaMalloc(&ws_ptr, workspace_size));
  }

  void* x_ptr;
  void* f_ptr;
  void* y_ptr;
  void (*init_fn)(void** d_ptr, size_t n, std::function<float()> init_fn);
  void (*print_fn)(void* d_ptr, size_t n, const std::string& prompt);
  if (opts.data_type == 0) {
    init_fn = InitDeviceTensor<float>;
    print_fn = PrintDeviceTensor<float>;
  } else {
    init_fn = InitDeviceTensor<__half>;
    print_fn = PrintDeviceTensor<__half>;
  }

  init_fn(&x_ptr, opts.input_size(), InitRandoms);
  init_fn(&f_ptr, opts.filter_size(), InitRandoms);
  init_fn(&y_ptr, opts.output_size(), InitRandoms);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(x_ptr, opts.input_size(), "### Input Before:");
    print_fn(f_ptr, opts.filter_size(), "### Filter Before:");
    print_fn(y_ptr, opts.output_size(), "### Output Before:");
  }




  std::array<void*, 3> data_ptrs = {x_ptr, y_ptr, f_ptr};
  std::array<int64_t, 3> uids = {'x', 'y', 'w'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
                         .setWorkspacePointer(ws_ptr)
                         .setDataPointers(data_ptrs.size(), data_ptrs.data())
                         .setUids(uids.size(), uids.data())
                         .build();
  checkCUDNN(variantPack.get_status());

  auto ret = cudnnBackendExecute(cudnn, plan_desc, variantPack.get_raw_desc());
  checkCUDNN(ret);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(x_ptr, opts.input_size(), "### Input After:");
    print_fn(f_ptr, opts.filter_size(), "### Filter After:");
    print_fn(y_ptr, opts.output_size(), "### Output After:");
  }
  std::cout << ">>> Convolution Finished." << std::endl;
}
