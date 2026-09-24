# Restickify: DMA requests and transpose throughput

The [cost model](cost_model.md) prices proven DL16 transports using their physical
source access, including stick-axis swaps and staging copies. A core split can make each input burst shorter
without reducing total payload, so total cores and aggregate bytes alone cannot
rank these plans.

## Geometry and hardware limits

The extractor uses the input's device layout and per-invocation coordinates,
including stick-plane/element coordinates. It coalesces adjacent affine axes
from stride one outward. A split on an inner axis prevents coalescing its outer
neighbors. Physical padding does too. Unsupported non-affine accesses, missing
ownership and non-DL16 layouts retain the previous model.

Off-chip loads transfer 128-byte words, with a maximum burst of 32 words
(4096 bytes). These are hardware transfer limits, not calibrated tile dimensions.
For payload `P`, source run `R`, and `C` active cores:

```text
requests = P / clamp(R, 128, 4096)
byte_time = P / bw_restickify
transfer_time = byte_time + max(byte_time, requests * ns_per_request(C))
transpose_time = P / (C * per_core_transpose_bandwidth)
extra = max(0, max(transfer_time, transpose_time) - 2 * byte_time)
```

The existing model already charges balanced-copy bandwidth. Add only `extra`,
multiplied by the operation's loop trip count. Add it outside bundle compute
overlap: the swap uses the on-chip transpose pipeline and precedes its dependent consumer.
This formula describes an HBM-to-HBM stick-axis swap. A plain copy or an HBM-to-LX
staging copy receives only the read-request excess over its existing byte charge,
without the transpose ceiling. An LX-resident input receives no off-chip request
charge. Symbolic residency preserves these rules while the solver chooses a plan.

Staging matters: the planner can see a copy feeding a transpose, while the final
program folds that copy into a direct off-chip read by the transpose. The copy's
division may differ from the consumer's, so charging the former does not price
the executed read. The objective uses a non-mutating view of the direct read
only when the existing address, ownership and loop checks prove copy removal
valid for **every candidate division**. It then prices the consumer's source
geometry and omits the removed copy. A failed proof or shared copy preserves the
original cost view. Allocation still plans the original buffers; the late pass
remains responsible for validating and performing the actual rewrite.

The implementation folds payload into the request-count expression **before**
CP-SAT integerization. Keeping requests/byte as an intermediate can silently round
the entire request term to zero. Solver tests check actual choices and objective
values, not just the presence of split symbols.

## Calibration, September 24, 2026

Controlled operation replay on Spyre 1.0, DL16. No planner
choices are involved. Input device order `[N,B,X/64,64]`, output
`[B,N/64,X,64]`; the same access as Granite's repeated key restickify. Every
configuration is checked bit-exactly against CPU. Device durations are medians of
three profiler kernel events, separately from 11 synchronized host timings.

At `B=8, X=128, N=1024` (2 MiB payload):

| Split B×X | Input run | Lowered input burst | Single invocation | 128 repetitions, per invocation |
|---|---:|---:|---:|---:|
| 4×1 | 512 B | 4 words | 51.06 us | 50.45 us |
| 8×1 | 256 B | 2 words | 82.72 us | 81.21 us |
| 4×2 | 128 B | 1 word | 143.76 us | 142.56 us |
| 8×2 | 128 B | 1 word | 144.61 us | 142.03 us |

Inspection of the generated programs confirms these burst lengths. This is a request-rate
effect already present **outside** loops. `4×1` is faster than `8×1`, not slower.

Varying `N` independently, with 32 advancing input tiles repeated four times:

| N | 4×1 | 8×1 | 4×2 | 8×2 |
|---:|---:|---:|---:|---:|
| 512 | 25.22 us | 40.70 us | 70.92 us | 72.61 us |
| 1024 | 50.63 us | 78.62 us | 142.84 us | 142.77 us |
| 2048 | 101.93 us | 163.57 us | 286.48 us | 286.63 us |

Reusing one input tile versus advancing through the KV backing does not materially
change these rates. The extra time scales with payload, not an exact-size gate.

An independent low-core sweep at `N=1024`, 32 repetitions, measured per invocation:
`1×1`: 54.03 us, `2×1`: 39.14 us, `4×1`: 51.03 us, `2×2`: 142.94 us.
The single-core result constrains the transpose pipeline ceiling (40 GB/s payload).

The sustained aggregate request intervals are **empirical hardware calibration**:
8.75 ns at two cores, 7.5 ns at 4/8/16 cores, 3.75 ns at 32 cores. One core uses
the two-core interval as a conservative fallback; its measurement is dominated
by the separately modelled transpose ceiling and cannot identify a request rate.
The 4–16 plateau and 32-core improvement are measured, but their microarchitectural
cause is not established. Do not interpret the table as a topology specification.

Cross-geometry check: at `B=8,X=256,N=512` (still 2 MiB), `8×1` measures 50.82 us,
`8×2` 81.67 us, `8×4` 76.81 us. This changes both run length and request parallelism;
it cannot be represented by an outer/inner split penalty independent of geometry.

A copy-only control preserves the stick axis and uses the same source access.
At `B=8,X=128,N=1024`, 32 repetitions, per invocation: `2×1` takes 34.30 us,
`4×1` 49.11 us, `8×1` 82.32 us, `4×2` 153.81 us and `8×2` 155.28 us.
The request-related slowdown is therefore not unique to the transpose pipeline.
The model captures these split-dependent increments within 15%; it does not
recalibrate the existing plain-copy bandwidth and read/write turnaround estimate.

## Scope and remaining validation

- No exact-size, repetition or preferred-split gate. Payload is the work visited
  per invocation, not the size of the KV backing allocation.
- DL16 source DMA calibration. Fully local transports and other device formats
  remain unchanged. HBM-to-LX uses the same source-request estimate; that staging
  case is not independently calibrated for absolute latency.
- Non-affine/sub-stick accesses are not calibrated. Unsupported core counts are
  neutral rather than assigned an invented measured rate.
- Output burst fragmentation and interactions between independent operations in
  a fused bundle are not separately modelled by this term.
- A frozen-plan one-block Granite 32k experiment confirms the direction of the
  split effects, but later samples developed runtime noise and a memory-mapping diagnostic.
  It is not evidence for a new stable end-to-end speedup; clean validation is required.

Use `examples/bench_restickify_dma.py` for controlled measurements. Hardware timing belongs in the
calibration report, not in deterministic unit-test timing assertions.
