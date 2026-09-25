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

"""D2H views must transfer their covered sticks, including offset and padding."""

import json

import pytest
import torch

import torch_spyre  # noqa: F401
from torch_spyre._C import DataFormats, SpyreTensorLayout


@pytest.fixture(scope="module", autouse=True)
def initialize_spyre_runtime():
    # Explicit-layout allocation needs the device allocator initialized first.
    torch.spyre._lazy_init()


def assert_d2h_bytes(prof, tmp_path, expected):
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    copies = [
        e
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"]
    ]
    assert copies, "The profiler must record the actual device transfer"
    assert sum(e["args"]["bytes"] for e in copies) == expected


@pytest.mark.parametrize(
    "dtype,device_format",
    [
        (torch.float16, DataFormats.SEN169_FP16),
        (torch.bfloat16, DataFormats.SEN169_FP16),
        (torch.int64, DataFormats.IEEE_INT32),
        (torch.float32, DataFormats.IEEE_FP32),
    ],
)
@pytest.mark.parametrize("width", [128, 130])
@pytest.mark.parametrize("row", [0, 3, 7])
def test_row_view_transfers_only_covered_sticks(
    dtype, device_format, width, row, tmp_path
):
    rows = 8
    # A period coprime to the row widths makes wrong-row reads observable.
    host = ((torch.arange(rows * width) % 251) - 125).to(dtype).reshape(1, rows, width)
    eps = device_format.elems_per_stick()
    sticks = (width + eps - 1) // eps
    # The row-major physical layout produced by the compiled LM head.
    layout = SpyreTensorLayout(
        [rows, sticks, 1, eps], [width, eps, -1, 1], device_format
    )
    device = host.to("spyre", device_layout=layout)
    view = device[:, row, :]
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = view.cpu()
    torch.testing.assert_close(actual, host[:, row, :], rtol=0, atol=0)
    assert_d2h_bytes(prof, tmp_path, sticks * 128)


@pytest.mark.parametrize(
    "dtype,device_format",
    [
        (torch.float16, DataFormats.SEN169_FP16),
        (torch.float32, DataFormats.IEEE_FP32),
    ],
)
@pytest.mark.parametrize("start,stop", [(5, 31), (64, 128), (5, 67), (65, 130)])
def test_partial_stick_view_transfer(dtype, device_format, start, stop, tmp_path):
    rows, width = 8, 130
    host = ((torch.arange(rows * width) % 251) - 125).to(dtype).reshape(rows, width)
    eps = device_format.elems_per_stick()
    layout = SpyreTensorLayout(
        [rows, (width + eps - 1) // eps, eps], [width, eps, 1], device_format
    )
    device = host.to("spyre", device_layout=layout)
    # Offsets within one stick and aligned multi-stick slices can use DCI
    # directly. Crossing a tile from an unaligned start uses the existing
    # full-allocation CPU staging fallback in spyre_copy_from.
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = device[3, start:stop].cpu()
    torch.testing.assert_close(actual, host[3, start:stop], rtol=0, atol=0)
    if start % eps and start % eps + stop - start > eps:
        expected_bytes = rows * ((width + eps - 1) // eps) * 128
    else:
        expected_bytes = ((stop + eps - 1) // eps - start // eps) * 128
    assert_d2h_bytes(prof, tmp_path, expected_bytes)


def test_strided_batch_slice_and_destination_dtype(tmp_path):
    host = (
        ((torch.arange(2 * 8 * 128) % 251) - 125).to(torch.float16).reshape(2, 8, 128)
    )
    layout = SpyreTensorLayout(
        [2, 8, 2, 1, 64], [1024, 128, 64, -1, 1], DataFormats.SEN169_FP16
    )
    device = host.to("spyre", device_layout=layout)
    # There is a gap between the selected rows. The bounded DMA must preserve
    # the original source strides when rebasing, and use the device item size.
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = device[:, 3, :].to(device="cpu", dtype=torch.float32)
    torch.testing.assert_close(actual, host[:, 3, :].float(), rtol=0, atol=0)
    assert_d2h_bytes(prof, tmp_path, 9 * 128 * 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_slice_to_larger_host_dtype(dtype, tmp_path):
    host = ((torch.arange(8 * 128) % 251) - 125).to(dtype).reshape(8, 128)
    layout = SpyreTensorLayout([8, 2, 64], [128, 64, 1], DataFormats.SEN169_FP16)
    device = host.to("spyre", device_layout=layout)
    # The 512-byte destination needs more capacity than the 256-byte DMA.
    # Flex must receive the CPU storage capacity, not the staging buffer size.
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = device[3].to(device="cpu", dtype=torch.float32)
    torch.testing.assert_close(actual, host[3].float(), rtol=0, atol=0)
    assert_d2h_bytes(prof, tmp_path, 256)


@pytest.mark.parametrize(
    "dtype,device_format",
    [(torch.float32, DataFormats.BFLOAT16), (torch.int32, DataFormats.IEEE_INT64)],
)
def test_other_storage_formats_keep_full_transfer(dtype, device_format, tmp_path):
    host = ((torch.arange(8 * 128) % 251) - 125).to(dtype).reshape(8, 128)
    eps = device_format.elems_per_stick()
    layout = SpyreTensorLayout([8, 128 // eps, eps], [128, eps, 1], device_format)
    device = host.to("spyre", device_layout=layout)
    torch.spyre.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]
    ) as prof:
        actual = device[3].cpu()
    torch.testing.assert_close(actual, host[3], rtol=0, atol=0)
    # Spyre storage reserves at least the logical CPU byte size, even when
    # the explicitly requested device encoding uses fewer bytes per element.
    assert_d2h_bytes(prof, tmp_path, max(host.nbytes, 8 * (128 // eps) * 128))
