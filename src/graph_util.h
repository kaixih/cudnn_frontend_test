#ifndef CUDNN_FE_TEST_SRC_GRAPH_UTIL_H_
#define CUDNN_FE_TEST_SRC_GRAPH_UTIL_H_

#include <cudnn_frontend.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

struct Edge {
  std::optional<std::string> tensor_name;
  std::optional<cudnn_frontend::Tensor*> tensor_ptr;
  Edge() {}
  Edge(const char* c) : tensor_name(c) {}
  Edge(cudnn_frontend::Tensor* t) : tensor_ptr(t) {}
};

struct Node {
  std::string op_name;
  std::string node_name;
  cudnnDataType_t op_dtype;
  cudnn_frontend::BackendDescriptor* desc;
  std::vector<double> scales;
  std::unordered_map<std::string, Edge> edges;
};

std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> CreateOpGraph(
    cudnnHandle_t& cudnn, std::vector<Node>& nodes);

std::optional<cudnn_frontend::Tensor> CreateCudnnTensor(
    const int64_t* dims, const int64_t* strides, int n, int64_t uid, int dtype,
    bool is_virtual = false);

#endif  // CUDNN_FE_TEST_SRC_GRAPH_UTIL_H_
