# SDPA KV-block = stick-rounded ¼ length + 8k prefill tests — design

**Date:** 2026-08-12
**Branch:** `flash-attn-kv-loop` (PR #3672 family)
**File touched:** `torch_spyre/_inductor/decompositions.py`
(`spyre__sdpa_overrideable`), `tests/inductor/test_inductor_ops.py`
(the `("test_sdpa", "test_sdpa_cpu")` param_sets).
**Predecessor handoff:** `docs/superpowers/plans/2026-08-11-sdpa-lq-tile-next-steps.md`

## Problem

The flash-attention SDPA decomp tiles the query axis (`max_seqlen_q`) and
loops over KV blocks of a fixed size (`kv_block_size = 64`,
`decompositions.py:441`). Fixed 64 means the block *count* grows with KV
length: `num_kv_blocks = ceil(Skv / 64)`. At an 8192-length KV that is 128
blocks.

PR #3672 review (mudhakar, 2026-08-12) reports **block counts > 6 do not yet
compile**; the PR's own tests exercise only 2–3 blocks. The one *verified*
8192 configuration from #3674 used **4 × 2048 chunks** — i.e. block size =
KV_length / 4, which pins the block count near 4 regardless of sequence
length.

Goal: reach a working state at 8k by sizing the KV block at ~¼ of the KV
length, and add 8k prefill tests that prove numeric correctness against the
CPU reference.

## Prerequisite discovered during exploration — stale C++ build

Branch HEAD `e1202fc` is a merge with `main` that pulled in commit `36af10b`
("provenance: link Spyre profiler events to debug handles", #3354), which
added `register_kernel_provenance` to `torch_spyre/csrc/module.cpp`. The
committed `_C.so` on disk predates that commit, so **both** baseline prefill
tests currently fail with:

```text
ImportError: cannot import name 'register_kernel_provenance'
from 'torch_spyre._C'
```

This is a Python↔C++ ABI mismatch from a stale build, **not** an SDPA
regression. `make setup`
(`uv sync --all-extras --active --inexact --reinstall-package torch-spyre`)
rebuilds the extension. This must complete and the baseline must return green
*before* any SDPA change is evaluated — otherwise a green/red signal is
meaningless. (Consistent with the `torch-spyre-main-tracks-upstream` memory:
the C++ build breaks when main moves.)

## Scope

**In scope**

- Change `kv_block_size` from fixed 64 to a stick-rounded ¼-of-KV formula.
- Add `mha_prefill_8k` and `gqa_prefill_8k` non-causal, no-mask test cases.
- Verify both on-device against the CPU reference; keep the existing baseline
  green.

**Explicitly out of scope (deferred)**

- Causal / additive-mask prefill (`coarse_tile: hint_id appears in more than
  one group`) — plan avenue #1.
- `kv_tail` partial last block (Skv not a multiple of the block size) — plan
  avenue #2.
- Activating the `batch_size` / `num_heads` tiles (the `_b`/`_h`
  placeholders) — plan avenue #3.
- Removing the `if True:` dead scaffolding — plan avenue #4.

These stay failing / as-is and are not touched by this effort.

## Design

### 1. Block-size formula

Replace `decompositions.py:441`:

```python
kv_block_size = 64
```

with:

```python
# Fixed block size of 64 gave num_blocks = ceil(Skv/64), which climbs with
# sequence length — 8k KV = 128 blocks, and block counts > 6 do not yet
# compile (PR #3672). Size the block at ~1/4 of the KV length instead, so
# the block *count* stays ~4 regardless of Skv (8k -> 2048, 4 blocks).
# Round up to a 64-element fp16 stick so every block stays stick-aligned;
# floor at 64 so short KV keeps the original small-block behavior.
kv_block_size = max(64, ((max_seqlen_kv + 3) // 4 + 63) // 64 * 64)
```

`num_kv_blocks`, the `for blk` loop, and `end = min(start + kv_block_size,
max_seqlen_kv)` already derive from `kv_block_size`, so this is the only
decomp edit.

Resulting shapes:

| `max_seqlen_kv` | `kv_block_size` | blocks | note |
|---|---|---|---|
| 8192 | 2048 | 4 | matches verified 4×2048 |
| 512 | 128 | 4 | |
| 128 | 64 | 2 | **unchanged from fixed-64** |
| 130 | 64 | 64,64,2 | tail unchanged (deferred) |

Because `ceil(128/4) = 32` floors back up to 64, every currently-passing
small test keeps its exact present block layout. The change is a **no-op**
for everything except long KV — so it cannot regress the green baseline.

### 2. 8k test cases

Add to the `("test_sdpa", "test_sdpa_cpu")` param_sets
(`test_inductor_ops.py:3082`), mirroring the verified shape
(`B=1, H=8, Lq=512, Lk=8192, D=128`) with the same `transpose(1, 2)` view the
existing cases use:

```python
"mha_prefill_8k": (
    cached_randn((1, 512, 8, 128), differentiation=1, dtype=torch.float16).transpose(1, 2),
    cached_randn((1, 8192, 8, 128), differentiation=2, dtype=torch.float16).transpose(1, 2),
    cached_randn((1, 8192, 8, 128), differentiation=3, dtype=torch.float16).transpose(1, 2),
    None, False, False,
),
"gqa_prefill_8k": (
    cached_randn((1, 512, 8, 128),  differentiation=1, dtype=torch.float16).transpose(1, 2),
    cached_randn((1, 8192, 2, 128), differentiation=2, dtype=torch.float16).transpose(1, 2),
    cached_randn((1, 8192, 2, 128), differentiation=3, dtype=torch.float16).transpose(1, 2),
    None, False, True,
),
```

Both non-causal, no attn mask. Correctness is checked by the framework's
`compare_with_cpu` against the `test_sdpa_cpu` reference — no separate
assertion. If a case passes it is **not** added to `expect_fail`; if it
surprises us we diagnose rather than mask.

### 3. Verification (on-device)

Runs directly on this Spyre-capable machine (`import torch` gives device
access; `torch.spyre.is_available()` is `True`).

1. Rebuild finishes → run baseline
   (`test_sdpa and (mha_prefill or gqa_prefill) and not causal and not
   kv_tail and not mask`). Must be green — proves the failures were the stale
   build.
2. Apply §1, re-run baseline. Must stay green (block size is a no-op there).
3. Apply §2, run the two 8k cases targeted (~46s+ compile each). Iterate
   to green on-device.
4. `pre-commit run --files` on the two changed files.
5. Commit `-s -S` only once the targeted tests are green.

## Testing

- Baseline SDPA prefill (existing): regression guard, must stay green through
  §1.
- `mha_prefill_8k`, `gqa_prefill_8k` (new): numeric proof of the ¼-block path
  at 128-block-equivalent sequence length, via `compare_with_cpu`.

## Risks

- **8k surfaces new codegen paths.** Larger tiles / more blocks than the small
  tests may hit untested backend code (bundle plan, layout). Mitigated by
  matching mudhakar's already-verified shape rather than inventing one.
- **Compile time.** ~46s+ per 8k case; run targeted, never the whole suite in
  a tight loop.
- **Rebuild time / cache.** ccache is warm (~83% cacheable); first rebuild
  after the merge may still be slow.

## Ground rules

- Every commit `git commit -s -S` (DCO + GPG); verify `echo test | gpg
  --clearsign` first. Never commit unsigned. Don't commit until targeted tests
  are green.
- `import regex` not `import re`; line length 88; `pre-commit run --files`
  before pushing.
- `main` must track `upstream/main` (missing `flex::`/ABI symbols otherwise) —
  see `torch-spyre-main-tracks-upstream`.

## Related memories

`torch-spyre-sdpa-lq-tile-accumulator`, `torch-spyre-sdpa-causal-alignment`,
`torch-spyre-main-tracks-upstream`, `torch-spyre-sdpa-torch-full`.
