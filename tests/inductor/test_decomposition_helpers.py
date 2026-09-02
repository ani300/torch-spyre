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

import pytest
import torch  # noqa: F401 - load Torch before its Spyre backend module

from torch_spyre._inductor.decompositions import (
    _kv_blocks_per_loop_group,
    _num_tiles_for_max_extent,
)


@pytest.mark.parametrize(
    ("sequence_length", "max_extent", "expected"),
    (
        (8192, 2048, 4),
        (32768, 2048, 16),
        (10, 4, 5),
    ),
)
def test_num_tiles_for_max_extent(sequence_length, max_extent, expected):
    assert _num_tiles_for_max_extent(sequence_length, max_extent) == expected


@pytest.mark.parametrize(
    ("num_q_tiles", "num_kv_blocks", "expected"),
    (
        (4, 4, 4),
        (4, 16, 4),
        (16, 16, 1),
        (1, 16, 16),
    ),
)
def test_kv_blocks_per_loop_group(num_q_tiles, num_kv_blocks, expected):
    assert _kv_blocks_per_loop_group(num_q_tiles, num_kv_blocks) == expected
