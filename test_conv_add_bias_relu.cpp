#include <cuda_fp16.h>

#include <iostream>

#include "cmd_options.h"
#include "common.h"
#include "graph_conv_add_bias_relu.h"
#include "utils.h"

int main(int argc, char** argv) {
  ASSIGN_OR_RETURN(auto opts, ParseConvOpts(argc, argv),
                   "Failed to parse the conv parameters.")
  bool print_on = false;
  const char* env_p = std::getenv("PRINTALL");
  if (env_p && (strcmp(env_p, "1") == 0 || strcmp(env_p, "true") == 0)) {
    print_on = true;
  }

  if (opts.bias_size() == 0) {
    std::cout << "--bias_dims has to be provided in this test!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  PrintConvOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));
  ASSIGN_OR_RETURN(auto op_graph, GetConvAddBiasReluGraph(opts, cudnn),
                   "Failed to build the conv graph.");
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
  void* b_ptr;
  void* y_ptr;
  void (*init_fn)(void** d_ptr, size_t n, std::function<float(int)> init_fn);
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
  init_fn(&b_ptr, opts.bias_size(), InitRandoms);
  init_fn(&y_ptr, opts.output_size(), InitRandoms);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(x_ptr, opts.input_size(), "### Input Before:");
    print_fn(f_ptr, opts.filter_size(), "### Filter Before:");
    print_fn(b_ptr, opts.bias_size(), "### Bias Before:");
  }

  int64_t uids[] = {'x', 'w', 'z', 'b', 'y'};
  auto launcher = LaunchRunner<void*, void*, void*, void*, void*>();
  launcher(cudnn, plan_desc, ws_ptr, uids, x_ptr, f_ptr, y_ptr, b_ptr, y_ptr);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(y_ptr, opts.output_size(), "### Output After:");
  }
  std::cout << ">>> Convolution Finished." << std::endl;
}
