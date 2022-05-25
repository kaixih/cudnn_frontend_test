#include <cudnn_frontend.h>

#include "cudnn_utils.h"

std::optional<cudnn_frontend::Tensor> CreateCudnnTensor(
    const int64_t* dims, const int64_t* strides, int n, int64_t uid, int dtype,
    bool is_virtual = false) {
  auto tensor = cudnn_frontend::TensorBuilder()
                    .setDim(n, dims)
                    .setStrides(n, strides)
                    .setId(uid)
                    .setAlignment(32)
                    .setDataType(ToCudnnDataType(dtype))
                    .setVirtual(is_virtual)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(tensor);
  return tensor;
}

const json* CudnnExecutionPlanEngineFilterStatic() {
  static std::string filter_str = R"({
      "version" : 1,
        "rules"   : [
          { "rule_id"             : "ConvFwd_eng999",
            "operation"           : "ConvFwd",
            "engine"              : 999,
            "knob"                : [],
            "cudnn_version_start" : 8000,
            "cudnn_version_end"   : -1
          }
      ]})";
  static const json* json_handle = new json(json::parse(filter_str));
  return json_handle;
}

const json* CudnnExecutionPlanEngineFilterRuntime() {
  static const json* json_handle = []() -> const json* {
    json j;
    if (cudnn_frontend::load_from_config(j, "")) {
      return new json(j);
    }
    return nullptr;
  }();
  return json_handle;
}

bool GenericEngineFilter(cudnnBackendDescriptor_t engine_config,
                         bool disable_winograd, bool disable_nondeterminism,
                         bool disable_tensor_core) {
  bool ret = cudnn_frontend::hasNumericalNote<
      CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);

  if (disable_winograd) {
    ret |= cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_WINOGRAD>(
        engine_config);
  }

  if (disable_nondeterminism) {
    ret |=
        cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
            engine_config);
  }

  if (disable_tensor_core) {
    ret |= cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(
        engine_config);
  }

  return ret;
}

void DisplayNumericNotes(
    cudnn_frontend::ManagedOpaqueDescriptor& engine_config) {
  cudnnBackendDescriptor_t engine_config_desc =
      engine_config->get_backend_descriptor();
  cudnn_frontend::ManagedOpaqueDescriptor engine =
      cudnn_frontend::make_shared_backend_pointer(
          CUDNN_BACKEND_ENGINE_DESCRIPTOR);
  cudnnBackendDescriptor_t engine_desc = engine->get_backend_descriptor();
  int64_t engine_count = -1;
  checkCUDNN(cudnnBackendGetAttribute(
      engine_config_desc, CUDNN_ATTR_ENGINECFG_ENGINE,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_count, &engine_desc));

  cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
  int64_t elem_count = 0;
  checkCUDNN(cudnnBackendGetAttribute(
      engine->get_backend_descriptor(), CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
      CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &elem_count,
      notes));
  if (elem_count != 0) {
    std::cout << "  Numeric Notes: ";
    for (int i = 0; i < elem_count; i++) {
      std::cout << CudnnStatusToString(notes[i]) << ", ";
    }
    std::cout << std::endl;
  }
}

void DisplayBehaviorNotes(
    cudnn_frontend::ManagedOpaqueDescriptor& engine_config) {
  cudnnBackendDescriptor_t engine_config_desc =
      engine_config->get_backend_descriptor();
  cudnn_frontend::ManagedOpaqueDescriptor engine =
      cudnn_frontend::make_shared_backend_pointer(
          CUDNN_BACKEND_ENGINE_DESCRIPTOR);
  cudnnBackendDescriptor_t engine_desc = engine->get_backend_descriptor();
  int64_t engine_count = -1;
  checkCUDNN(cudnnBackendGetAttribute(
      engine_config_desc, CUDNN_ATTR_ENGINECFG_ENGINE,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_count, &engine_desc));

  cudnnBackendBehaviorNote_t notes[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
  int64_t elem_count = 0;
  checkCUDNN(cudnnBackendGetAttribute(
      engine->get_backend_descriptor(), CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
      CUDNN_TYPE_BEHAVIOR_NOTE, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, &elem_count,
      notes));
  if (elem_count != 0) {
    std::cout << "  Behavior Notes: ";
    for (int i = 0; i < elem_count; i++) {
      std::cout << CudnnStatusToString(notes[i]) << ", ";
    }
    std::cout << std::endl;
  }
}

void CreateOpRunners(
    cudnnHandle_t& cudnn,
    std::unique_ptr<cudnn_frontend::OperationGraph> op_graph,
    std::vector<std::unique_ptr<cudnn_frontend::ExecutionPlan>>* out_runners) {
  cudnn_frontend::EngineConfigList filtered_configs;
  auto generic_filter_fn = [=](cudnnBackendDescriptor_t engine_config) -> bool {
    return GenericEngineFilter(engine_config,
                               /*disable_winograd*/ false,
                               /*disable_nondeterminism*/ false,
                               /*disable_tensor_core*/ false);
  };

  // The runtime fusion engines will always be in the "heuristics_fallback". To
  // avoid early-exit, we set the evaluate_all to true. 
  std::vector<cudnnStatus_t> heuristics_statuses =
      cudnn_frontend::get_heuristics_list<2>(
          {"heuristics_mode_b", "heuristics_fallback"}, *op_graph,
          generic_filter_fn, filtered_configs, /*evaluate_all=*/true);
  std::cout << "\nFiltered engine configs size: " << filtered_configs.size()
            << std::endl;

  auto fn = []() { return true; };
  auto maybe_json_handle_static = CudnnExecutionPlanEngineFilterStatic();
  auto maybe_json_handle_runtime = CudnnExecutionPlanEngineFilterRuntime();

  out_runners->clear();
  for (int i = 0; i < filtered_configs.size(); i++) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnn)
                    .setEngineConfig(filtered_configs[i], op_graph->getTag())
                    .build();
    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
      continue;
    }

    if (maybe_json_handle_static &&
        cudnn_frontend::check_errata(*maybe_json_handle_static, plan.getTag(),
                                     cudnn, fn)) {
      std::cout << "Exclude engine (static): " << plan.getTag() << std::endl;
      continue;
    }
    if (maybe_json_handle_runtime &&
        cudnn_frontend::check_errata(*maybe_json_handle_runtime, plan.getTag(),
                                     cudnn, fn)) {
      std::cout << "Exclude engine (runtime): " << plan.getTag() << std::endl;
      continue;
    }

    std::cout << "Adding engine (" << out_runners->size()
              << "): " << plan.getTag() << std::endl;

    DisplayNumericNotes(filtered_configs[i]);
    DisplayBehaviorNotes(filtered_configs[i]);
    if (plan.getWorkspaceSize()) {
      std::cout << "  Workspace Bytes: " << plan.getWorkspaceSize()
                << std::endl;
    }

    out_runners->push_back(std::unique_ptr<cudnn_frontend::ExecutionPlan>(
        new cudnn_frontend::ExecutionPlan(std::move(plan))));
  }

  std::cout << "\nReturned execution plans size: " << out_runners->size()
            << std::endl;
}

template <typename... Args>
struct LaunchRunner {
  void operator()(cudnnHandle_t& cudnn, cudnnBackendDescriptor_t& plan_desc,
                  void* ws_ptr, const int64_t* uids, Args... args) {
    std::array<void*, sizeof...(Args)> data_ptrs = {args...};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(ws_ptr)
                           .setDataPointers(data_ptrs.size(), data_ptrs.data())
                           .setUids(data_ptrs.size(), uids)
                           .build();
    checkCUDNN(variantPack.get_status());

    auto ret =
        cudnnBackendExecute(cudnn, plan_desc, variantPack.get_raw_desc());
    checkCUDNN(ret);
  }
};
