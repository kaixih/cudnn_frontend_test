#include "graph_runner.h"

#include <sys/time.h>
#include <optional>

namespace {
uint64_t CpuTimer() {
  timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

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
    printf("  Numeric Notes: ");
    for (int i = 0; i < elem_count; i++) {
      printf("%s, ", CudnnNumericalNoteToString(notes[i]).c_str());
    }
    printf("\n");
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
    printf("  Behavior Notes: ");
    for (int i = 0; i < elem_count; i++) {
      printf("%s, ", CudnnBehaviorNoteToString(notes[i]).c_str());
    }
    printf("\n");
  }
}
}  // namespace

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
          generic_filter_fn, filtered_configs, /*evaluate_all=*/false);
  for (auto& status : heuristics_statuses) {
    if (status != CUDNN_STATUS_SUCCESS) {
      // The mode_b is supposed to fallback to mode_a if it cannot support
      // things. We need this feature esp. for the runtime fusion engines.
      // However, there is a known issue for this. So, we manually do that here.
      // TODO(kaixih): fix this when cudnn fixes it.
      heuristics_statuses = cudnn_frontend::get_heuristics_list<2>(
          {"heuristics_mode_a", "heuristics_fallback"}, *op_graph,
          generic_filter_fn, filtered_configs, /*evaluate_all=*/false);
      for (auto& status : heuristics_statuses) {
        if (status != CUDNN_STATUS_SUCCESS) {
          printf(RED "!!! cuDNN's get_heuristics_list error\n" RESET);
          return;
        }
      }
    }
  }

  printf("\nFiltered engine configs size: %ld\n", filtered_configs.size());

  auto fn = []() { return true; };
  auto maybe_json_handle_static = CudnnExecutionPlanEngineFilterStatic();
  auto maybe_json_handle_runtime = CudnnExecutionPlanEngineFilterRuntime();

  out_runners->clear();
  for (int i = 0; i < filtered_configs.size(); i++) {
    uint64_t t0 = CpuTimer();
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnn)
                    .setEngineConfig(filtered_configs[i], op_graph->getTag())
                    .build();
    uint64_t t1 = CpuTimer();
    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
      continue;
    }
    printf("Compilation time (ms): %lu\n", t1 - t0);

    if (maybe_json_handle_static &&
        cudnn_frontend::check_errata(*maybe_json_handle_static, plan.getTag(),
                                     cudnn, fn)) {
      printf("Exclude engine (static): %s\n", plan.getTag().c_str());
      continue;
    }
    if (maybe_json_handle_runtime &&
        cudnn_frontend::check_errata(*maybe_json_handle_runtime, plan.getTag(),
                                     cudnn, fn)) {
      printf("Exclude engine (runtime): %s\n", plan.getTag().c_str());
      continue;
    }

    printf("Adding engine (%ld): %s\n", out_runners->size(),
           plan.getTag().c_str());

    DisplayNumericNotes(filtered_configs[i]);
    DisplayBehaviorNotes(filtered_configs[i]);
    if (plan.getWorkspaceSize()) {
      printf("  Workspace Bytes: %ld\n", plan.getWorkspaceSize());
    }

    out_runners->push_back(std::unique_ptr<cudnn_frontend::ExecutionPlan>(
        new cudnn_frontend::ExecutionPlan(std::move(plan))));
  }

  printf("\nReturned execution plans size: %ld\n", out_runners->size());
}
