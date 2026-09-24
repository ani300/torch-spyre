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


"""Controlled DMA geometry replay, with stick-swap and copy-only controls."""

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
import torch_spyre
from sympy import Integer, Mod, Symbol, floor
from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout
from torch_spyre._inductor.op_spec import OpSpec, TensorArg, LoopSpec
from torch_spyre.execution.async_compile import _compile_to_dir, _run_backend_compiler
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sizes", default="1024")
    ap.add_argument("--trips", default="1,128")
    ap.add_argument("--splits", default="8x1,8x2,4x1,4x2")
    ap.add_argument("--x", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--backing", type=int, default=33280)
    ap.add_argument(
        "--advance",
        type=int,
        default=0,
        help="Number of distinct input tiles (0=reuse)",
    )
    ap.add_argument("--samples", type=int, default=11)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--copy-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    b, x = args.batch, args.x
    torch.manual_seed(101)
    cases = [
        (int(n), int(t), *(int(s) for s in split.split("x")))
        for n in args.sizes.split(",")
        for t in args.trips.split(",")
        for split in args.splits.split(",")
    ]
    if b <= 0 or x <= 0 or x % 64 or args.advance < 0 or args.samples <= 0:
        ap.error(
            "batch and samples must be positive; X must be a positive multiple of 64; advance cannot be negative"
        )
    for n, trips, sb, sx in cases:
        if n <= 0 or n % 64 or trips <= 0 or sb <= 0 or sx <= 0:
            ap.error(
                "N must be a positive multiple of 64; trips and splits must be positive"
            )
        if b % sb or x % sx or sb * sx > 32:
            ap.error("splits must divide B and X and use at most 32 cores")
        if args.advance and trips % args.advance:
            ap.error("trips must be divisible by advance")
    random.Random(101).shuffle(cases)
    print(
        json.dumps({"environment": torch_spyre.__file__, "args": vars(args)}),
        flush=True,
    )
    for n, trips, sb, sx in cases:
        kind = "copy" if args.copy_only else "swap"
        name = f"{kind}_b{b}_x{x}_n{n}_t{trips}_s{sb}x{sx}_a{args.advance}_back{args.backing}"
        outdir = root / name
        outdir.mkdir(exist_ok=True)
        backing = max(args.backing, n * max(1, args.advance))
        c0, c1, c2, core, tile = [
            Symbol(s) for s in ("c0", "c1", "c2", "core_id", "tile")
        ]
        inp = TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            [backing, b, x // 64, 64],
            [c2, c0, floor(c1 / 64), Mod(c1, 64)],
            allocation={"hbm": 0},
            device_tile_advance_expr=Integer(n * b * x) * tile
            if args.advance
            else None,
        )
        out = TensorArg(
            False,
            1,
            DataFormats.SEN169_FP16,
            [b, n // 64, x, 64],
            [c0, floor(c2 / 64), c1, Mod(c2, 64)],
            allocation={"hbm": 1},
        )
        op = OpSpec(
            "ReStickifyOpHBM",
            False,
            {c0: (Integer(b), sb), c1: (Integer(x), sx), c2: (Integer(n), 1)},
            [inp, out],
            {},
            tiled_symbols=[[tile], []] if args.advance else [[]] if trips > 1 else [],
            tiled_symbol_trip_counts={tile: args.advance} if args.advance else {},
            core_id_to_work_slice={
                c0: Mod(core, sb),
                c1: Mod(floor(core / sb), sx),
                c2: Integer(0),
            },
        )
        if args.copy_only:
            op.op = "identity"
            out.device_size = [b, n, x // 64, 64]
            out.device_coordinates = [c0, c2, floor(c1 / 64), Mod(c1, 64)]
        if args.advance:
            assert trips % args.advance == 0
            specs = [
                LoopSpec(
                    Integer(trips // args.advance),
                    [LoopSpec(Integer(args.advance), [op])],
                )
            ]
        else:
            specs = [LoopSpec(Integer(trips), [op])] if trips > 1 else [op]
        start = time.perf_counter()
        symbols = _compile_to_dir(name, str(outdir), specs, 0)
        _run_backend_compiler(name, str(outdir), dict(os.environ))
        compile_s = time.perf_counter() - start
        runner = SpyreSDSCKernelRunner(name, str(outdir), symbol_kinds=symbols)
        cpu_in = torch.randint(-16, 17, (backing, b, x)).to(torch.float16)
        in_layout = SpyreTensorLayout(
            device_size=[backing, b, x // 64, 64],
            stride_map=[b * x, x, 64, 1],
            device_dtype=DataFormats.SEN169_FP16,
        )
        in_dev = cpu_in.to(device="spyre", device_layout=in_layout)
        out_layout = SpyreTensorLayout(
            device_size=out.device_size,
            stride_map=[x * n, x, 64, 1] if args.copy_only else [x * n, 64, n, 1],
            device_dtype=DataFormats.SEN169_FP16,
        )
        out_dev = spyre_empty_with_layout(
            (b, n, x) if args.copy_only else (b, x, n),
            (x * n, x, 1) if args.copy_only else (x * n, n, 1),
            torch.float16,
            out_layout,
            device=torch.device("spyre"),
        )
        for _ in range(3):
            runner.run(in_dev, out_dev)
        torch.spyre.synchronize()
        offset = n * max(0, args.advance - 1)
        order = (1, 0, 2) if args.copy_only else (1, 2, 0)
        expected = cpu_in[offset : offset + n].permute(order)
        torch.testing.assert_close(out_dev.cpu(), expected, rtol=0, atol=0)
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter_ns()
            runner.run(in_dev, out_dev)
            torch.spyre.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        record = dict(
            name=name,
            b=b,
            x=x,
            n=n,
            trips=trips,
            sb=sb,
            sx=sx,
            backing=backing,
            advance=args.advance,
            compile_s=compile_s,
            us=samples,
            median_us=statistics.median(samples),
            host_us_per_iteration=statistics.median(samples) / trips,
            correct=True,
            copy_only=args.copy_only,
        )
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ]
            ) as prof:
                for _ in range(3):
                    runner.run(in_dev, out_dev)
                torch.spyre.synchronize()
            trace_path = outdir / "trace.json"
            prof.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
            device_us = [e["dur"] for e in events if e.get("cat") == "kernel"]
            if device_us:
                record["device_us_per_iteration"] = statistics.median(device_us) / trips
        print(json.dumps(record), flush=True)
        with (root / "results.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
