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

"""BF16 public tensors must communicate their existing DL16 device storage.

Run with torchrun --nproc-per-node 4 -m pytest
tests/distributed/test_bfloat16_collectives.py.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_spyre  # noqa: F401
from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout

if "RANK" not in os.environ or int(os.getenv("WORLD_SIZE", "0")) < 2:
    pytest.skip("Requires torchrun with at least two ranks", allow_module_level=True)

DEVICE = torch.device(f"spyre:{os.environ['RANK']}")
GROUP_NAME = "bf16_collective_tests"


@pytest.fixture(scope="module", autouse=True)
def process_group():
    torch.spyre._impl._lazy_init()
    dist.init_process_group("cpu:gloo,spyre:spyreccl")
    c10d._register_process_group(GROUP_NAME, dist.group.WORLD)
    yield
    dist.destroy_process_group()


def rank_input(rank, shape):
    # Powers of two are exact in BF16 and DL16, including values above the
    # IEEE FP16 range. Nonuniform signed values detect raw-format mistakes.
    pattern = torch.tensor(
        [2.0**20, -(2.0**20), 2.0**-20, -(2.0**-20), 0.5, -0.5, 1.5, -1.5],
        dtype=torch.bfloat16,
    )
    return (pattern.repeat(math.prod(shape) // len(pattern)) * (rank + 1)).reshape(
        shape
    )


def expected_result(op, shape):
    world = dist.get_world_size()
    if op == "broadcast":
        return rank_input(world - 1, shape)
    values = [rank_input(rank, shape) for rank in range(world)]
    if op == "allgather":
        return torch.cat(values, dim=0)
    return torch.stack(values).sum(dim=0)


@pytest.mark.parametrize("op", ["allreduce", "reduce", "broadcast", "allgather"])
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize("shape", [(128,), (4, 64)])
def test_bfloat16_eager(op, async_op, shape):
    x = rank_input(dist.get_rank(), shape).to(DEVICE)
    if op == "allreduce":
        work = dist.all_reduce(x, async_op=async_op)
    elif op == "reduce":
        work = dist.reduce(x, dst=dist.get_world_size() - 1, async_op=async_op)
    elif op == "broadcast":
        work = dist.broadcast(x, src=dist.get_world_size() - 1, async_op=async_op)
    else:
        outputs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        work = dist.all_gather(outputs, x, async_op=async_op)
    if async_op:
        assert work is not None
        work.wait()
    else:
        assert work is None
    if op == "reduce" and dist.get_rank() != dist.get_world_size() - 1:
        return
    result = (
        torch.cat([part.cpu() for part in outputs]) if op == "allgather" else x.cpu()
    )
    torch.testing.assert_close(result, expected_result(op, shape), rtol=0, atol=0)


@pytest.mark.parametrize("op", ["allreduce", "broadcast", "allgather"])
@pytest.mark.parametrize("shape", [(128,), (4, 96)])
def test_bfloat16_compiled(op, shape):
    world = dist.get_world_size()

    def fn(x):
        x = x * 2.0
        if op == "allreduce":
            y = torch.ops._c10d_functional.all_reduce(x, "sum", GROUP_NAME)
        elif op == "broadcast":
            y = torch.ops._c10d_functional.broadcast(x, world - 1, GROUP_NAME)
        else:
            y = torch.ops._c10d_functional.all_gather_into_tensor(x, world, GROUP_NAME)
        return torch.ops._c10d_functional.wait_tensor(y) * 0.5

    torch.compiler.reset()
    compiled = torch.compile(fn, fullgraph=True)
    for _ in range(2):
        x = rank_input(dist.get_rank(), shape).to(DEVICE)
        result = compiled(x)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(
            result.cpu(), expected_result(op, shape), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "op",
    [
        "allreduce",
        "reduce",
        "broadcast",
        "allgather",
        "allreduce_run",
        "broadcast_run",
        "allgather_run",
    ],
)
def test_bfloat16_rejects_non_dl16_storage(op):
    # Expert layouts can store actual BF16 bits while retaining the logical
    # torch.bfloat16 dtype. Never reinterpret those bits as DL16 arithmetic.
    layout = SpyreTensorLayout([2, 64], [64, 1], DataFormats.BFLOAT16, {(0, 1): 128})
    x = spyre_empty_with_layout([128], [1], torch.bfloat16, layout, device=DEVICE)
    with pytest.raises(
        RuntimeError, match="BF16 collectives require DL16 device storage"
    ):
        if op == "allreduce":
            dist.all_reduce(x)
        elif op == "reduce":
            dist.reduce(x, dst=0)
        elif op == "broadcast":
            dist.broadcast(x, src=0)
        elif op == "allgather":
            outputs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
        elif op == "allreduce_run":
            torch.ops.spyre.allreduce_run(x, -1)
        elif op == "broadcast_run":
            torch.ops.spyre.broadcast_run(x, -1, 0)
        else:
            torch.ops.spyre.allgather_run(x, -1, dist.get_world_size())
