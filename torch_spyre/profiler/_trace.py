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

"""Chrome trace export with explicit CPU launch to Spyre submission edges."""

import gzip
import json
from collections import defaultdict
from pathlib import Path

_FLOW_CATEGORY = "spyre_launch_to_submit"


def _add_submission_flows(trace):
    # Own category and IDs: preserve unrelated CUDA/CPU flows, and be idempotent.
    events = [e for e in trace["traceEvents"] if e.get("cat") != _FLOW_CATEGORY]
    cpu = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in (
            "cpu_op",
            "user_annotation",
        ):
            external_id = event.get("args", {}).get("External id")
            if external_id:
                cpu[event["pid"], external_id].append(event)

    flows = []
    missing = 0
    for event in events:
        if event.get("ph") != "X" or event.get("name") != "aiuSubmitToHardware":
            continue
        args = event.get("args", {})
        external_id = args.get("External id", args.get("external_correlation"))
        sources = cpu.get((event["pid"], external_id), ())
        # Do not invent a link by time proximity if attribution is missing or
        # ambiguous (e.g. a producer reused IDs when merging profiling sessions).
        if len(sources) != 1:
            missing += 1
            continue
        source = sources[0]
        flow_id = len(flows) // 2 + 1
        metadata = {"external_id": external_id, "batch": args.get("correlation")}
        for endpoint, phase in ((source, "s"), (event, "f")):
            flow = {
                "name": "CPU launch → Spyre submit",
                "cat": _FLOW_CATEGORY,
                "ph": phase,
                "id": flow_id,
                "pid": endpoint["pid"],
                "tid": endpoint["tid"],
                "ts": endpoint["ts"],
                "args": metadata,
            }
            if phase == "f":
                flow["bp"] = "e"
            flows.append(flow)
    trace["traceEvents"] = events + flows
    trace["spyre_submission_links"] = {
        "linked_submissions": len(flows) // 2,
        "submissions_without_unique_cpu_source": missing,
    }


def export_chrome_trace(prof, path):
    """Export a torch profiler trace with CPU → submit and hardware-launch → device navigation.

    The native profiler supplies external correlation metadata and hardware-launch-to-
    device flows. This adds a distinct CPU-to-submit edge for every submission,
    including when one CPU operation produces multiple hardware batches. No
    activity timestamps or durations are changed. JSON and JSON.gz are supported.
    """
    path = Path(path)
    prof.export_chrome_trace(str(path))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    _add_submission_flows(trace)
    with opener(path, "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, ensure_ascii=False)
