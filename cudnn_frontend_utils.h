#include <cudnn_frontend.h>

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

  std::vector<cudnnStatus_t> heuristics_statuses =
      cudnn_frontend::get_heuristics_list<2>(
          {"heuristics_mode_b", "heuristics_fallback"}, *op_graph,
          generic_filter_fn, filtered_configs);
  for (auto& status : heuristics_statuses) {
    if (!status == CUDNN_STATUS_SUCCESS) {
      std::cout << "!!! cuDNN's get_heuristics_list error." << std::endl;
      return;
    }
  }
  std::cout << "Filtered engine configs size: " << filtered_configs.size()
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
    auto numeric_notes = plan.getNumericNotes();
    std::vector<std::string> set_numeric_notes;
    for (int i = 0; i < numeric_notes.size(); i++) {
      if (numeric_notes[i]) {
        set_numeric_notes.push_back(
            CudnnStatusToString(static_cast<cudnnBackendNumericalNote_t>(i)));
      }
    }
    if (!set_numeric_notes.empty()) {
      std::cout << "  Numeric Notes: ";
      for (auto& note : set_numeric_notes) {
        std::cout << note << ", ";
      }
      std::cout << std::endl;
    }

    auto behavior_notes = plan.getBehaviorNotes();
    std::vector<std::string> set_behavior_notes;
    for (int i = 0; i < behavior_notes.size(); i++) {
      if (behavior_notes[i]) {
        set_behavior_notes.push_back(
            CudnnStatusToString(static_cast<cudnnBackendBehaviorNote_t>(i)));
      }
    }
    if (!set_behavior_notes.empty()) {
      std::cout << "  Behavior Notes: ";
      for (auto& note : set_behavior_notes) {
        std::cout << note << ", ";
      }
      std::cout << std::endl;
    }

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
