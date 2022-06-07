#include <cuda_fp16.h>

#include <iostream>

#include "cmd_options.h"
#include "common.h"
#include "graph_matmul_bias_tanh.h"
#include "utils.h"

int main(int argc, char** argv) {
  ASSIGN_OR_RETURN(auto opts, ParseMatMulOpts(argc, argv),
                   "Failed to parse the matmul parameters.")
  bool print_on = false;
  const char* env_p = std::getenv("PRINTALL");
  if (env_p && (strcmp(env_p, "1") == 0 || strcmp(env_p, "true") == 0)) {
    print_on = true;
  }

  if (opts.bias_size() == 0) {
    std::cout << "--bias_dims has to be provided in this test!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  PrintMatMulOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));
  ASSIGN_OR_RETURN(auto op_graph, GetMatMulBiasTanhGraph(opts, cudnn),
                   "Failed to build the MatMulBiasTanh graph.");
  std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>> plans;
  CreateOpRunners(cudnn, std::move(op_graph), &plans);

  int engine_index = ParseEngineOpts(argc, argv);
  if (engine_index >= plans.size()) {
    std::cout << "!!! invalid engine_index: " << engine_index
              << " is out of range of [0, " << plans.size() << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::cout << "Using (" << engine_index
            << "): " << plans[engine_index]->getTag() << std::endl;
  auto plan_desc = plans[engine_index]->get_raw_desc();
  auto workspace_size = plans[engine_index]->getWorkspaceSize();
  void* ws_ptr = nullptr;
  if (workspace_size != 0) {
    checkCUDA(cudaMalloc(&ws_ptr, workspace_size));
  }

  void* a_ptr;
  void* b_ptr;
  void* c_ptr;
  void* z_ptr;
  void (*init_fn)(void** d_ptr, size_t n, std::function<float()> init_fn);
  void (*print_fn)(void* d_ptr, size_t n, const std::string& prompt);
  if (opts.data_type == 0) {
    init_fn = InitDeviceTensor<float>;
    print_fn = PrintDeviceTensor<float>;
  } else {
    init_fn = InitDeviceTensor<__half>;
    print_fn = PrintDeviceTensor<__half>;
  }

  init_fn(&a_ptr, opts.input0_size(), InitRandoms);
  init_fn(&b_ptr, opts.input1_size(), InitRandoms);
  init_fn(&c_ptr, opts.output_size(), InitRandoms);
  init_fn(&z_ptr, opts.bias_size(), InitRandoms);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(a_ptr, opts.input0_size(), "### InputA Before:");
    print_fn(b_ptr, opts.input1_size(), "### InputB Before:");
    print_fn(z_ptr, opts.bias_size(), "### Bias Before:");
  }

  int64_t uids[] = {'a', 'b', 'c', 'z'};
  auto launcher = LaunchRunner<void*, void*, void*, void*>();
  launcher(cudnn, plan_desc, ws_ptr, uids, a_ptr, b_ptr, c_ptr, z_ptr);

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(c_ptr, opts.output_size(), "### Output After:");
  }
  std::cout << ">>> MatMul Finished." << std::endl;
}
