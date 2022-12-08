# CUDNN Frontend API Samples
[cuDNN frontend/graph APIs](https://github.com/NVIDIA/cudnn-frontend) allows
users to input an operation graph and under the hood it will look for efficient
engines to execute the graph.

Using such APIs usually involves many boilerplate codes and requires the
understanding of the engine charateristics, since not all returned engines meet
our requirements on precision/determinism/etc. A typical procedure is like:

1. Define the operation graph.
1. Obtain the list of heuristics engines.
1. Pick up one engine based on your requirements (e.g. fastest, tensor core,
   deterministic.).
1. Prepare input data.
1. Execute the engine.

This repo contains a scalable code framework that we've already used in the
Tensorflow and allows easily adding new operation graphs. At this point, we
support two primary graphs: convolution fusion graphs and matmul fusion graphs
which are roughly based on the pattern `[Conv|MatMul]+Bias+Activation` (listed
[here](src/graph_builder.h)). 

# Usage
## Install
```bash
$ make run_conv_graphs.out
$ make run_matmul_graphs.out
```
Note, the repo requires the cuDNN 8.5+ and is tested with cuDNN frontend v0.7.

## Supported Graphs:
You can find the most updated list of supported graphs in the
[graph_builder.h](src/graph_builder.h). Currently the following graph patterns
are supported:
```cpp
enum class GraphType {
  ConvFwd = 0,
  ConvBwdFilter = 1,
  ConvBwdData = 2,
  ConvAddBiasRelu = 3,
  ConvBiasElu = 4,
  ConvBiasRelu6 = 5,
  ConvBiasLeakyRelu = 6,
  ConvBn = 7,

  MatMulBiasTanh = 100,
  MatMulBiasSigmoid = 101,
  MatMulBiasGeluExact = 102
};
```

## Run Convolution Graphs
The example below displays the `Conv-Bias-Elu` graph is executed with specified
convolution shapes, the fp16 inputs/outputs, and the channels_last data format.
In addition, we'd like to use the 2nd returned engine from the heuristics list.
At the end of the output, the execution time of the selected engine is also
collected.

```
$ ./run_conv_graphs.out -input 8,64,128,128 -filter 32,64,3,3 -bias 1,32,1,1 -data_type 1 -data_format 1 -graph_index 4 -engine_index 2
>>> Retrieved CONVOLUTION specs:
>>>   num_spatial_dims: 2,
>>>   input_dims (-input): 8, 64, 128, 128,
>>>   filter_dims (-filter): 32, 64, 3, 3,
>>>   bias_dims (-bias): 1, 32, 1, 1,
>>>   output_dims (-output): 8, 32, 128, 128,
>>>   input_strides: 1048576, 1, 8192, 64,
>>>   filter_strides: 576, 1, 192, 64,
>>>   bias_strides: 32, 1, 32, 32,
>>>   output_strides: 524288, 1, 4096, 32,
>>>   paddings (-padding): 1, 1,
>>>   strides (-stride): 1, 1,
>>>   dilations (-dilation): 1, 1,
>>>   data_type (-data_type [0<fp32>|1<fp16>]): 1,
>>>   data_format (-data_format [0<nchw>|1<nhwc>]): 1,
>>>   engine_index (-engine_index): 2,
>>>   graph_index (-graph_index <int>(+100 for matmul graphs)): 4
>>>   graph_name: ConvBiasEluGraph
...
Using (2): ConvFwd_Add_EluFwd_eng0_k24=0
Execution time(ms): 0.288051
>>> Convolution Finished.
```
## Run MatMul Graphs
Similarly, the following example shows a matmul graph of
`MatMul-Bias-GeluExact`.
```
$ ./run_matmul_graphs.out -a 1,8,16 -b 1,16,32 -bias 1,1,32 -data_type 1 -data_format 1 -engine_index 4 -graph_index 102
>>> Retrieved MatMul specs:
>>>   num_dims: 3,
>>>   a_dims (-a): 1, 8, 16,
>>>   b_dims (-b): 1, 16, 32,
>>>   bias_dims (-bias): 1, 1, 32,
>>>   c_dims (-c): 1, 8, 32,
>>>   a_strides: 128, 16, 1,
>>>   b_strides: 512, 32, 1,
>>>   bias_strides: 32, 32, 1,
>>>   c_strides: 256, 32, 1,
>>>   transpose_a (-transpose_a): 0,
>>>   transpose_b (-transpose_b): 0,
>>>   data_type (-data_type [0<fp32>|1<fp16>]): 1,
>>>   engine_index (-engine_index): 4,
>>>   graph_index (-graph_index <int>(+100 for matmul graphs)): 102
>>>   graph_name: MatMulBiasGeluExactGraph
...
Using (4): Matmul_Add_GeluFwd_eng0_k24=5
Execution time(ms): 0.006963
>>> MatMul Finished.
```


# Graph Representation
Typically, users need to manually build the edges ("virtual tensors") and nodes
("operations") for the fusion graph. But this might be verbose and error-prone.
In this repo, we support a list based graph representation (below) and our
utility functions can analyze the list to build the cuDNN virtual tensors and
operations and wire them together accordingly.

```c++
{
  {"op1", op1_dtype, &op1_desc, {/*scaling factors*/}, {{"x", &tensor_x}, {"y", ""}},
  {"op2", op2_dtype, &op2_desc, {/*scaling factors*/}, {{"x", "op1:y"}, {"y", ""}}},
  // ...
}
```

Note, this feature is still in the experimental stage. For simplicity, we made
certain assumptions and ignored certain details what will be improved in the
future.
* The virtual tensor dtype is inferred from the corresponding op. Ideally, its
  dtype should be consistent with the subsequent ops to avoid unnecessary type
  conversion.
* The virtual tensor's output shape is simply copied from the real output tensor
  and we assume there is only one output tensor in the graph. Ideally, the
  output shape of each virtual tensor should be inferred from its corresponding
  input tensor and operation.

