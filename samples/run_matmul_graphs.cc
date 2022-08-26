#include <cuda_fp16.h>

#include "graph_builder.h"
#include "graph_runner.h"
#include "util.h"

int main(int argc, char** argv) {
  auto opts = ParseMatMulOpts(argc, argv);
  PrintMatMulOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));

  int graph_index = opts.graph_index;
  PrintGraphName(graph_index);
  ASSIGN_OR_RETURN(auto graph_builder,
                   GetMatMulGraphBuilderByIndex(graph_index),
                   "Failed to find the valid graph builder.");

  ASSIGN_OR_RETURN(auto op_graph, graph_builder(opts, cudnn),
                   "Failed to build the graph.");

  std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>> plans;
  CreateOpRunners(cudnn, std::move(op_graph), &plans);

  int engine_index = opts.engine_index;
  if (engine_index >= plans.size()) {
    printf(RED "!!! Invalid engine index: %d\n" RESET, engine_index);
    std::exit(EXIT_FAILURE);
  }
  printf("Using (%d): %s\n", engine_index,
         plans[engine_index]->getTag().c_str());

  auto plan_desc = plans[engine_index]->get_raw_desc();
  auto workspace_size = plans[engine_index]->getWorkspaceSize();
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    checkCUDA(cudaMalloc(&workspace_ptr, workspace_size));
  }

  void (*init_fn)(void** d_ptr, size_t n, std::function<float(int)> init_fn);
  void (*print_fn)(void* d_ptr, size_t n, const std::string& prompt);
  if (opts.data_type == 0) {
    init_fn = AllocTensorWithInitValues<float>;
    print_fn = PrintTensor<float>;
  } else {
    init_fn = AllocTensorWithInitValues<__half>;
    print_fn = PrintTensor<__half>;
  }

  void* a_ptr;
  void* b_ptr;
  void* c_ptr;
  void* z_ptr;

  init_fn(&a_ptr, opts.a_size(), InitRandoms);
  init_fn(&b_ptr, opts.b_size(), InitRandoms);
  init_fn(&c_ptr, opts.c_size(), InitRandoms);
  init_fn(&z_ptr, opts.bias_size(), InitRandoms);
  checkCUDA(cudaDeviceSynchronize());

  bool print_on = IsPrintAll();
  if (print_on) {
    print_fn(a_ptr, opts.a_size(), "### matrix_a:");
    print_fn(b_ptr, opts.b_size(), "### matrix_b:");
    print_fn(z_ptr, opts.bias_size(), "### bias:");
  }

  switch (graph_index) {
    case GraphType::MatMulBiasTanh:
    case GraphType::MatMulBiasSigmoid:
    case GraphType::MatMulBiasGeluExact: {
      int64_t uids[] = {'a', 'b', 'c', 'z'};
      auto launcher = LaunchOpRunner<void*, void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, a_ptr, b_ptr, c_ptr,
               z_ptr);
      break;
    }
    default: {
      printf(RED "!!! Unsupported graph index: %d\n" RESET, graph_index);
      std::exit(EXIT_FAILURE);
    }
  }

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(c_ptr, opts.c_size(), "### matrix_c:");
  }
  printf(GREEN ">>> MatMul Finished.\n" RESET);
}
