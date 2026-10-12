# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise ordinary torch.profiler export through the native Spyre session."""

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def native_traces(tmp_path_factory):
    if os.environ.get("USE_SPYRE_PROFILER", "1") == "0":
        pytest.skip("Spyre profiler support is disabled")
    if not os.environ.get("RUNTIME_INSTALL_DIR"):
        pytest.skip("Requires the native build environment (RUNTIME_INSTALL_DIR)")
    output = tmp_path_factory.mktemp("native_spyre_trace")
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", MAX_JOBS="4")
    # An isolated process avoids backend auto-init and scopes the synthetic
    # Kineto child profiler to these CPU-only tests.
    subprocess.run(
        [sys.executable, __file__, str(output)], env=env, check=True, timeout=300
    )
    return output


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


@pytest.mark.parametrize(
    "filename",
    [
        "trace.json",
        "trace.json.gz",
        "repeat.json",
        "scheduled-2.json",
        "scheduled-4.json",
    ],
)
def test_standard_export_contains_both_flow_stages(native_traces, filename):
    trace = _read(native_traces / filename)
    events = trace["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    cpu = next(event for event in spans if event["name"] == "test_cpu_origin")
    markers = [event for event in spans if event["name"] == "aiuCpuLaunch"]
    submits = [event for event in spans if event["name"] == "aiuSubmitToHardware"]
    launches = [event for event in spans if event["name"] == "aiuLaunchControlBlocks"]
    kernels = [event for event in spans if event["name"] == "synthetic_kernel"]
    assert len(markers) == len(launches) == len(kernels) == 2
    assert len(submits) == 6
    assert all(event["dur"] == 5 for event in submits)
    assert all(event["dur"] == 2 for event in kernels)
    assert all(event["dur"] == 0 for event in markers)
    assert all(
        (event["pid"], event["tid"], event["ts"]) == (cpu["pid"], cpu["tid"], cpu["ts"])
        for event in markers
    )
    assert {event["args"]["correlation"] for event in markers} == {7, 8}
    flows = {}
    for event in events:
        if event.get("ph") in ("s", "f") and event.get("cat") == "ac2g":
            flows.setdefault(event["id"], []).append(event)
    assert len(flows) == 4
    for flow_id, endpoints in flows.items():
        assert len(endpoints) == 2
        start = next(event for event in endpoints if event["ph"] == "s")
        finish = next(event for event in endpoints if event["ph"] == "f")
        assert start["ts"] <= finish["ts"]
        source_spans, target_spans = (
            (markers, submits) if flow_id > 2**32 else (launches, kernels)
        )
        for endpoint, candidates in ((start, source_spans), (finish, target_spans)):
            assert any(
                (endpoint["pid"], endpoint["tid"], endpoint["ts"])
                == (candidate["pid"], candidate["tid"], candidate["ts"])
                for candidate in candidates
            )
    # Wide, missing, cross-process and calibration origins produce no CPU flow.
    wide = next(event for event in submits if event["args"]["correlation"] == 10)
    assert wide["args"]["external_correlation"] > 2**40
    assert "External id" not in wide["args"]
    assert "spyre_submission_links" not in trace


def _capture_native_traces(output):
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    csrc = root / "torch_spyre/csrc"
    runtime = Path(os.environ["RUNTIME_INSTALL_DIR"])
    aiupti = Path(os.environ.get("LIBAIUPTI_INSTALL_DIR") or runtime)
    external = Path(os.environ.get("SHARED_DEPS_DIR", csrc / "external"))
    fixture = load(
        name="spyre_native_trace_fixture",
        sources=[
            str(Path(__file__).with_name("native_trace_fixture.cpp")),
            *(
                str(csrc / "profiler" / name)
                for name in (
                    "AiuptiActivityApi.cpp",
                    "AiuptiActivityProfiler.cpp",
                    "AiuptiActivityHandlers.cpp",
                )
            ),
            str(csrc / "kernel_provenance_registry.cpp"),
        ],
        extra_include_paths=[
            str(csrc),
            str(csrc / "profiler"),
            str(runtime / "include"),
            str(aiupti / "include"),
            str(external / "nlohmann/include"),
            str(Path(torch.__file__).parent / "include/kineto"),
        ],
        extra_cflags=[
            "-std=c++20",
            "-O0",
            "-DHAS_AIUPTI",
            "-DUSE_KINETO",
            "-DFMT_HEADER_ONLY",
        ],
        extra_ldflags=[
            f"-L{runtime / 'lib'}",
            "-lflex",
            f"-L{aiupti / 'lib'}",
            "-laiupti",
        ],
    )
    fixture.register_profiler()
    for filename in ("trace.json", "trace.json.gz", "repeat.json"):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            fixture.emit_batches()
        prof.export_chrome_trace(str(output / filename))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2),
        on_trace_ready=lambda p: p.export_chrome_trace(
            str(output / f"scheduled-{p.step_num}.json")
        ),
    ) as prof:
        for _ in range(4):
            if prof.current_action == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
                fixture.emit_batches()
            prof.step()


if __name__ == "__main__":
    _capture_native_traces(Path(sys.argv[1]))
