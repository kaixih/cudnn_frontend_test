#ifndef CUDNN_FE_TEST_SRC_GRAPH_RUNNER_H_
#define CUDNN_FE_TEST_SRC_GRAPH_RUNNER_H_

#include <cudnn_frontend.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "util.h"

void CreateOpRunners(
    cudnnHandle_t& cudnn,
    std::unique_ptr<cudnn_frontend::OperationGraph> op_graph,
    std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>>* out_runners);

template <typename... Args>
struct LaunchOpRunner {
  void operator()(cudnnHandle_t& cudnn, cudnnBackendDescriptor_t& plan_desc,
                  void* ws_ptr, const int64_t* uids, Args... args) {
    std::array<void*, sizeof...(Args)> data_ptrs = {args...};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(ws_ptr)
                           .setDataPointers(data_ptrs.size(), data_ptrs.data())
                           .setUids(data_ptrs.size(), uids)
                           .build();
    checkCUDNN(variantPack.get_status());

    auto cudnn_execute = [&](int steps) {
      for (int i = 0; i < steps; i++) {
        auto ret =
            cudnnBackendExecute(cudnn, plan_desc, variantPack.get_raw_desc());
        checkCUDNN(ret);
      }
    };

    int kWarmupCount = 10;
    int kBenchmarkCount = 10;
    bool print_on = IsPrintAll();
    if (print_on) {
      kWarmupCount = 0;
      kBenchmarkCount = 1;
    }

    cudnn_execute(kWarmupCount);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudnn_execute(kBenchmarkCount);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time(ms): %f\n", milliseconds / kBenchmarkCount);
  }
};

#endif  // CUDNN_FE_TEST_SRC_GRAPH_RUNNER_H_
