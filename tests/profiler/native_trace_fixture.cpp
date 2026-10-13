/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Synthetic AIUPTI records exercise the production session and Kineto exporter
// without initializing a device or collecting hardware activity.
#include <ATen/record_function.h>
#include <libaiupti/aiupti_runtime_cbid.h>
#include <pybind11/pybind11.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "AiuptiActivityApi.h"
#include "AiuptiActivityProfiler.h"

namespace {
using libkineto::ActivityType;
using libkineto::AiuptiActivityApi;
using libkineto::AiuptiActivityBufferDeque;
using libkineto::AiuptiActivityProfilerSession;
using libkineto::Config;
using libkineto::IActivityProfiler;
using libkineto::IActivityProfilerSession;
using libkineto::Pti_Activity;

class SyntheticApi : public AiuptiActivityApi {
 public:
  std::vector<AIUpti_ActivityAPI> runtime;
  std::vector<AIUpti_ActivityThread> threads;
  std::vector<AIUpti_ActivityCompute> kernels;

  std::unique_ptr<AiuptiActivityBufferDeque> activityBuffers() override {
    return std::make_unique<AiuptiActivityBufferDeque>();
  }

  const std::pair<int, int> processActivities(
      AiuptiActivityBufferDeque&,
      std::function<void(const Pti_Activity*)> handler) override {
    // Deliberately deliver device records before their CPU attribution.
    for (const auto& record : kernels) {
      handler(reinterpret_cast<const Pti_Activity*>(&record));
    }
    for (const auto& record : runtime) {
      handler(reinterpret_cast<const Pti_Activity*>(&record));
    }
    // Names arrive after their events, as can happen across separate queues.
    for (const auto& record : threads) {
      handler(reinterpret_cast<const Pti_Activity*>(&record));
    }
    return {runtime.size() + kernels.size() + threads.size(), 0};
  }
};

SyntheticApi syntheticApi;
thread_local std::vector<uint64_t> externalIds;

class SyntheticSession : public AiuptiActivityProfilerSession {
 public:
  SyntheticSession(const Config& config, const std::set<ActivityType>& types)
      : AiuptiActivityProfilerSession(syntheticApi, config, types) {}

  void start() override {
    syntheticApi.runtime.clear();
    syntheticApi.threads.clear();
    syntheticApi.kernels.clear();
    externalIds.clear();
  }
  void stop() override {}
  void pushCorrelationId(uint64_t id) override {
    externalIds.push_back(id);
  }
  void popCorrelationId() override {
    externalIds.pop_back();
  }
  void pushUserCorrelationId(uint64_t) override {}
  void popUserCorrelationId() override {}
};

class SyntheticProfiler : public IActivityProfiler {
 public:
  const std::string& name() const override {
    return name_;
  }
  const std::set<ActivityType>& availableActivities() const override {
    return types_;
  }
  std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>&, const Config& config) override {
    return std::make_unique<SyntheticSession>(config, types_);
  }
  std::unique_ptr<IActivityProfilerSession> configure(
      int64_t, int64_t, const std::set<ActivityType>& types,
      const Config& config) override {
    return configure(types, config);
  }

 private:
  const std::string name_{"synthetic_spyre"};
  const std::set<ActivityType> types_{ActivityType::CPU_OP};
};

uint64_t emitBatch(uint32_t batch, const std::string& mode) {
  const uint64_t external = externalIds.empty() ? 0 : externalIds.back();
  const auto now = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  AIUpti_ActivityAPI submit{};
  submit.activity_kind = AIUPTI_ACTIVITY_KIND_RUNTIME;
  submit.cbid = AIUPTI_RUNTIME_TRACE_CBID_SUBMIT_TO_HARDWARE;
  submit.process_id = getpid();
  submit.thread_id = getpid() + 1000;  // A different scheduler lane.
  submit.start = now;
  submit.end = now + 5000;
  submit.correlation_id = batch;
  submit.external_id = external;
  // One thread intentionally has no metadata, exercising the numeric fallback.
  if (mode == "orphan" || batch == 8) ++submit.thread_id;
  if (syntheticApi.threads.empty()) {
    AIUpti_ActivityThread thread{};
    thread.activity_kind = AIUPTI_ACTIVITY_KIND_THREAD;
    thread.process_id = submit.process_id;
    thread.thread_id = submit.thread_id;
    std::memcpy(thread.name, "test-scheduler", sizeof("test-scheduler"));
    syntheticApi.threads.push_back(thread);
  }
  if (mode == "orphan") submit.external_id = 0;
  if (mode == "wide") submit.external_id |= uint64_t{1} << 40;
  if (mode == "calibration") submit.calibration_id = 1;
  if (mode == "other_process") ++submit.process_id;
  syntheticApi.runtime.push_back(submit);
  if (mode == "normal") {
    auto launch = submit;
    launch.cbid = AIUPTI_RUNTIME_TRACE_CBID_LAUNCH_CB_CMPT;
    launch.start = now + 1000;
    launch.end = now + 2000;
    syntheticApi.runtime.push_back(launch);
    AIUpti_ActivityCompute kernel{};
    kernel.activity_kind = AIUPTI_ACTIVITY_KIND_CMPT;
    kernel.operation_kind = AIUPTI_ACTIVITY_CMPT_KIND_EXEC;
    kernel.correlation_id = batch;
    kernel.start = now + 2000;
    kernel.end = now + 4000;
    std::memcpy(kernel.name, "synthetic_kernel", sizeof("synthetic_kernel"));
    syntheticApi.kernels.push_back(kernel);
  }
  return external;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("register_profiler", [] {
    libkineto::api().registerProfilerFactory(
        [] { return std::make_unique<SyntheticProfiler>(); });
  });
  module.def("emit_batches", [] {
    // Operator scopes use the external correlation stack, whereas Python's
    // record_function uses the separate user-correlation stack.
    at::RecordFunction origin(at::RecordScope::FUNCTION);
    origin.before("test_cpu_origin");
    uint32_t batch = 7;
    for (const auto* mode : {"normal", "normal", "orphan", "wide",
                             "calibration", "other_process"}) {
      TORCH_CHECK(emitBatch(batch++, mode) != 0,
                  "Expected an active CPU correlation ID");
    }
  });
}
