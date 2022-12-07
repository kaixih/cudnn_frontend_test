#include <cuda_fp16.h>

#include "graph_builder.h"
#include "graph_runner.h"
#include "util.h"

int main(int argc, char** argv) {
  auto opts = ParseConvOpts(argc, argv);
  PrintConvOpts(opts);

  cudnnHandle_t cudnn = nullptr;
  checkCUDNN(cudnnCreate(&cudnn));

  int graph_index = opts.graph_index;
  PrintGraphName(graph_index);
  ASSIGN_OR_RETURN(auto graph_builder, GetConvGraphBuilderByIndex(graph_index),
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
  void* f_ptr;
  void* b_ptr;
  void* y_ptr;
  void* mean_ptr;
  void* var_ptr;
  void* gamma_ptr;
  void* beta_ptr;

  init_fn(&x_ptr, opts.input_size(), InitRandoms);
  init_fn(&f_ptr, opts.filter_size(), InitRandoms);
  init_fn(&b_ptr, opts.bias_size(), InitRandoms);
  init_fn(&y_ptr, opts.output_size(), InitRandoms);
  init_fn(&mean_ptr, opts.filter_dims[0], InitZeros);
  init_fn(&var_ptr, opts.filter_dims[0], InitOnes);
  init_fn(&gamma_ptr, opts.filter_dims[0], InitRandoms);
  init_fn(&beta_ptr, opts.filter_dims[0], InitRandoms);
  checkCUDA(cudaDeviceSynchronize());

  bool print_on = IsPrintAll();
  if (print_on) {
    print_fn(x_ptr, opts.input_size(), "### input:");
    print_fn(f_ptr, opts.filter_size(), "### filter:");
    print_fn(b_ptr, opts.bias_size(), "### bias:");
    print_fn(mean_ptr, opts.filter_dims[0], "### mean:");
    print_fn(var_ptr, opts.filter_dims[0], "### var:");
    print_fn(gamma_ptr, opts.filter_dims[0], "### gamma:");
    print_fn(beta_ptr, opts.filter_dims[0], "### beta:");
  }

  auto graph_type = static_cast<GraphType>(graph_index);
  switch (graph_type) {
    case GraphType::ConvFwd:
    case GraphType::ConvBwdFilter:
    case GraphType::ConvBwdData: {
      int64_t uids[] = {'x', 'y', 'w'};
      auto launcher = LaunchOpRunner<void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, y_ptr, f_ptr);
      break;
    }
    case GraphType::ConvAddBiasRelu: {
      int64_t uids[] = {'x', 'w', 'z', 'b', 'y'};
      auto launcher = LaunchOpRunner<void*, void*, void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, f_ptr, y_ptr,
               b_ptr, y_ptr);
      break;
    }
    case GraphType::ConvBiasElu:
    case GraphType::ConvBiasRelu6:
    case GraphType::ConvBiasLeakyRelu: {
      int64_t uids[] = {'x', 'w', 'b', 'y'};
      auto launcher = LaunchOpRunner<void*, void*, void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, f_ptr, b_ptr,
               y_ptr);
      break;
    }
    case GraphType::ConvBn: {
      int64_t uids[] = {'x', 'w', 'y', 'm', 'v', 'g', 'b', 'e'};
      float epsilon = 0.001f;
      auto launcher = LaunchOpRunner<void*, void*, void*, void*, void*, void*,
                                     void*, void*>();
      launcher(cudnn, plan_desc, workspace_ptr, uids, x_ptr, f_ptr, y_ptr,
               mean_ptr, var_ptr, gamma_ptr, beta_ptr, &epsilon);
      break;
    }
    default: {
      printf(RED "!!! Unsupported graph index: %d\n" RESET, graph_index);
      std::exit(EXIT_FAILURE);
    }
  }

  checkCUDA(cudaDeviceSynchronize());
  if (print_on) {
    print_fn(y_ptr, opts.output_size(), "### output:");
  }
  printf(GREEN ">>> Convolution Finished.\n" RESET);
}
