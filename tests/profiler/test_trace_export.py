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

import copy
import gzip
import json

import pytest

from torch_spyre.profiler._trace import _add_submission_flows, export_chrome_trace


def _event(name, category, pid, tid, ts, external_id, batch=None):
    return {
        "ph": "X",
        "name": name,
        "cat": category,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": 2,
        "args": {"External id": external_id, "correlation": batch},
    }


def test_submission_flows_follow_ids_across_threads_and_batches():
    launch = _event("launch::H2D", "cpu_op", 10, 11, 100, 2**40)
    unrelated = _event("launch::Compute", "cpu_op", 10, 11, 119, 20)
    other_rank = _event("launch::H2D", "cpu_op", 20, 21, 90, 2**40)
    submit1 = _event(
        "aiuSubmitToHardware", "privateuse1_runtime", 10, 12, 120, 2**40, 1
    )
    submit2 = _event(
        "aiuSubmitToHardware", "privateuse1_runtime", 10, 12, 125, 2**40, 2
    )
    orphan = _event("aiuSubmitToHardware", "privateuse1_runtime", 10, 12, 130, 777, 3)
    native_flow = {"ph": "s", "cat": "ac2g", "name": "ac2g", "id": 1}
    original = [launch, unrelated, other_rank, submit1, submit2, orphan, native_flow]
    snapshot = copy.deepcopy(original)
    trace = {"traceEvents": original, "baseTimeNanoseconds": 42}
    _add_submission_flows(trace)
    assert trace["traceEvents"][: len(original)] == snapshot
    flows = trace["traceEvents"][len(original) :]
    assert [(e["ph"], e["ts"], e["tid"]) for e in flows] == [
        ("s", 100, 11),
        ("f", 120, 12),
        ("s", 100, 11),
        ("f", 125, 12),
    ]
    assert flows[0]["id"] == flows[1]["id"] != flows[2]["id"] == flows[3]["id"]
    assert trace["spyre_submission_links"]["submissions_without_unique_cpu_source"] == 1
    once = copy.deepcopy(trace)
    _add_submission_flows(trace)
    assert trace == once


def test_ambiguous_cpu_id_is_not_guessed():
    event = _event("launch::Compute", "cpu_op", 1, 2, 100, 7)
    submit = _event("aiuSubmitToHardware", "privateuse1_runtime", 1, 3, 120, 7, 9)
    trace = {"traceEvents": [event, copy.deepcopy(event), submit]}
    _add_submission_flows(trace)
    assert len(trace["traceEvents"]) == 3
    assert trace["spyre_submission_links"]["linked_submissions"] == 0


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_export_preserves_metadata_and_supports_compression(tmp_path, suffix):
    trace = {"traceEvents": [], "baseTimeNanoseconds": 1234, "deviceProperties": []}

    class Profiler:
        def export_chrome_trace(self, path):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "wt") as stream:
                json.dump(trace, stream)

    path = tmp_path / ("trace" + suffix)
    export_chrome_trace(Profiler(), path)
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(path, "rt") as stream:
        exported = json.load(stream)
    assert exported["baseTimeNanoseconds"] == 1234
    assert exported["deviceProperties"] == []
