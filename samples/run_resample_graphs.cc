#include <cuda_fp16.h>

#include "graph_builder.h"
#include "graph_runner.h"
#include "util.h"

int main(int argc, char** argv) {
  auto opts = ParseResampleOpts(argc, argv);
  PrintResampleOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));

  int graph_index = opts.graph_index;
  PrintGraphName(graph_index);
  ASSIGN_OR_RETURN(auto graph_builder,
                   GetResampleGraphBuilderByIndex(graph_index),
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

  void* x_ptr;
  void* y_ptr;
  void* idx_ptr;

  init_fn(&x_ptr, opts.input_size(), InitRandoms);
  init_fn(&y_ptr, opts.output_size(), InitRandoms);
  // Index tensor uses int8 dtype.
  AllocTensorWithInitValues<char>(&idx_ptr, opts.output_size(), InitOnes);
  checkCUDA(cudaDeviceSynchronize());

  auto graph_type = static_cast<GraphType>(graph_index);

  bool print_on = IsPrintAll();
  if (print_on) {
    if (graph_type == GraphType::AvgPoolFwd ||
        graph_type == GraphType::MaxPoolFwd) {
      print_fn(x_ptr, opts.input_size(), "### input (x):");
    } else {
      print_fn(y_ptr, opts.output_size(), "### input (dy):");
    }
    if (graph_type == GraphType::MaxPoolBwd) {
      PrintTensor<char>(idx_ptr, opts.output_size(), "### index:");
    }
  }

  switch (graph_type) {
    case GraphType::AvgPoolFwd: {
      int64_t uids[] = {'x', 'y'};
      auto launcher = LaunchOpRunner<void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, y_ptr);
      break;
    }
    case GraphType::AvgPoolBwd: {
      int64_t uids[] = {'x', 'y'};
      auto launcher = LaunchOpRunner<void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, y_ptr);
      break;
    }
    case GraphType::MaxPoolFwd: {
      int64_t uids[] = {'x', 'y', 'i'};
      auto launcher = LaunchOpRunner<void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, y_ptr, idx_ptr);
      break;
    }
    case GraphType::MaxPoolBwd: {
      int64_t uids[] = {'x', 'y', 'i'};
      auto launcher = LaunchOpRunner<void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, y_ptr, idx_ptr);
      break;
    }
    default: {
      printf(RED "!!! Unsupported graph index: %d\n" RESET, graph_index);
      std::exit(EXIT_FAILURE);
    }
  }

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    if (graph_type == GraphType::AvgPoolFwd ||
        graph_type == GraphType::MaxPoolFwd) {
      print_fn(y_ptr, opts.output_size(), "### output (y):");
    } else {
      print_fn(x_ptr, opts.input_size(), "### output (dx):");
    }

    if (graph_type == GraphType::MaxPoolFwd) {
      PrintTensor<char>(idx_ptr, opts.output_size(), "### index:");
    }
  }
  printf(GREEN ">>> Convolution Finished.\n" RESET);
}
