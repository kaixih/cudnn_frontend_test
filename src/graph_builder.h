#ifndef CUDNN_FE_TEST_SRC_GRAPH_BUILDER_H_
#define CUDNN_FE_TEST_SRC_GRAPH_BUILDER_H_

#include <cudnn_frontend.h>

#include <memory>
#include <optional>

#include "util.h"

enum class GraphType {
  ConvFwd = 0,
  ConvBwdFilter = 1,
  ConvBwdData = 2,
  ConvAddBiasRelu = 3,
  ConvBiasElu = 4,
  ConvBiasRelu6 = 5,
  ConvBiasLeakyRelu = 6,
  ConvBn = 7,

  MatMul = 100,
  MatMulBiasTanh = 101,
  MatMulBiasSigmoid = 102,
  MatMulBiasGeluExact = 103,

  AvgPoolFwd = 200,
  AvgPoolBwd = 201,
  MaxPoolFwd = 202,
  MaxPoolBwd = 203
};

typedef std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> (
    *ConvGraphBuilderFnPtr)(ConvOpts&, cudnnHandle_t&);

typedef std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> (
    *MatMulGraphBuilderFnPtr)(MatMulOpts&, cudnnHandle_t&);

typedef std::optional<std::unique_ptr<cudnn_frontend::OperationGraph>> (
    *ResampleGraphBuilderFnPtr)(ResampleOpts&, cudnnHandle_t&);

std::optional<ConvGraphBuilderFnPtr> GetConvGraphBuilderByIndex(
    int graph_index);
std::optional<MatMulGraphBuilderFnPtr> GetMatMulGraphBuilderByIndex(
    int graph_index);
std::optional<ResampleGraphBuilderFnPtr> GetResampleGraphBuilderByIndex(
    int graph_index);

void PrintGraphName(int graph_index);

#endif  // CUDNN_FE_TEST_SRC_GRAPH_BUILDER_H_
