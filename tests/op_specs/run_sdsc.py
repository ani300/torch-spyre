# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Run the paged-add SDSC through the torch-spyre pipeline.

Takes a pre-built sdsc_0.json (e.g. /tmp/paged_add_bundle/sdsc_0.json), copies
it into a fresh bundle directory with a bundle.mlir wrapper, invokes
dxp_standalone, then runs the compiled kernel against tensors sized for the
paged test's N_ = {out=128, mb=8, x=256, y=2}.

Dimension-order note
--------------------

SDSC layoutDimOrder_ and PyTorch shape are reversed. A tensor declared
[out=128, mb=8, x=256, y=2] in the SDSC corresponds to PyTorch shape
[y=2, x=256, mb=8, out=128]. For the paged test the stick dim is `out`,
which in PyTorch order is the innermost (last) dim — so the 4D tensors get
the default contiguous layout. The index tensor's SDSC layout [x=256, y=2]
becomes PyTorch [y=2, x=256]; its stick dim is `y`, which is the outermost
(first) PyTorch dim, so we build an explicit SpyreTensorLayout to place the
stick on dim 0.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import SpyreTensorLayout
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner


BUNDLE_MLIR = """module {
\tfunc.func @sdsc_bundle() {
\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}
\t\treturn
\t}
}
"""


def setup_bundle(sdsc_path, kernel_name):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    bundle_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    shutil.copyfile(sdsc_path, os.path.join(bundle_dir, "sdsc_0.json"))
    with open(os.path.join(bundle_dir, "bundle.mlir"), "w") as f:
        f.write(BUNDLE_MLIR)
    return bundle_dir


def allocate_and_move():
    """Allocate paged-gather tensors and move to spyre with explicit layouts.

    Dimension-order rule: SDSC layoutDimOrder_ and PyTorch shape are reversed.
    The stickification algorithm (spyre_tensor_impl.cpp:46) takes the host
    dim_order `[o0, o1, ..., o(N-1)]` and produces the device dim_map:
        rank 4: [o1, o2, o3, o0, o3]   (stick_dim = o3, stick_count at idx 3)
        rank 2: [o1, o0, o1]           (stick_dim = o1, stick_count at idx 0)

    Value tensor (the paged "table" being indexed into):
      SDSC   [out=128, mb=8, x=32, y=2], stickDim=out
      PyTorch [y=2, x=32, mb=8, out=128], stick dim → PyTorch dim 3 (default).

    Const scalar broadcast:
      1 fp16 scalar, broadcast over all logical dims.

    Index tensor (picks which `mb` row at each (x, y) point):
      SDSC   [x=32, y=2], stickDim=y
      PyTorch [y=2, x=32], stick dim → PyTorch dim 0.
      dim_order=[1, 0] places stick on PyTorch dim 0 (= y).

    Output tensor (uses the INDEX's dims: one fp16 per (x, y), per out):
      Logical shape: [out=128, x=32, y=2] — dropped `mb`, since the gather
      picks one mb-row per (x, y) pair. Implemented by the SDSC as a 4D
      [out, mb, x, y] tensor with mb constrained to 1 via maxDimSizes_[mb]=1.
      PyTorch shape: [y=2, x=32, mb=1, out=128] — stick on last dim (out).
    """
    torch.manual_seed(0xAFFE)

    # Value tensor: full [y=2, x=32, mb=8, out=128] fp16 table.
    value_host = torch.rand(2, 32, 8, 128, dtype=torch.float16)

    # Scalar broadcast constant.
    const_host = torch.zeros(1, dtype=torch.float16)

    # Output tensor: mb collapsed to 1 (gathered dim), carries the gathered
    # value + const. Size = y × x × mb × out = 2 × 32 × 1 × 128 = 8192 elems.
    out_host = torch.zeros(2, 32, 1, 128, dtype=torch.float16)

    # Index tensor: one index per (x, y); values pick an mb row in [0, 8).
    # int64 gets auto-downcast to int32 by the backend.
    index_host = torch.randint(0, 8, (2, 32), dtype=torch.int64)
    layout_idx = SpyreTensorLayout([2, 32], [32, 1], torch.int64, [1, 0])

    value_dev = value_host.to("spyre")
    const_dev = const_host.to("spyre")
    index_dev = index_host.to(device_layout=layout_idx)
    out_dev = out_host.to("spyre")

    print(f"value device_layout: {value_dev.device_tensor_layout()}")
    print(f"const device_layout: {const_dev.device_tensor_layout()}")
    print(f"index device_layout: {index_dev.device_tensor_layout()}")
    print(f"out   device_layout: {out_dev.device_tensor_layout()}")

    host = [value_host, const_host, index_host, out_host]
    dev = [value_dev, const_dev, index_dev, out_dev]
    return host, dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdsc_path", type=Path)
    parser.add_argument("--kernel-name", default="paged_run")
    parser.add_argument("--print", dest="do_print", action="store_true")
    args = parser.parse_args()

    if not args.sdsc_path.is_file():
        sys.exit(f"not a file: {args.sdsc_path}")

    bundle_dir = setup_bundle(args.sdsc_path, args.kernel_name)
    print(f"bundle_dir: {bundle_dir}")

    subprocess.run(
        ["dxp_standalone", "--bundle", "-d", bundle_dir],
        check=True,
    )

    host_tensors, dev_tensors = allocate_and_move()
    SpyreSDSCKernelRunner(args.kernel_name, bundle_dir).run(*dev_tensors)
    for host_t, dev_t in zip(host_tensors, dev_tensors):
        host_t[:] = dev_t.cpu()

    if args.do_print:
        for i, t in enumerate(host_tensors):
            print(f"--- tensor {i} ---")
            print(t)


if __name__ == "__main__":
    main()
