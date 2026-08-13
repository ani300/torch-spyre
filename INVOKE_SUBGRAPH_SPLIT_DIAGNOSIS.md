# invoke_subgraph × split_multi_ops — root cause & fix sketch

## Symptom

Compiling a model that uses `torch.compiler.nested_compile_region` (the
Granite whole-forward: one decoder block wrapped as a region, reused across
layers) aborts at **compile time** with:

```
torch._inductor.exc.InductorError: AssertionError: Node to insert before is not in graph.
  torch_spyre/_inductor/split_multi_ops.py:548  with gl.graph.inserting_before(orig_node):
  torch/fx/graph.py:1651                         raise AssertionError("Node to insert before is not in graph.")
```

Reproduced host-side with fake tensors (no VFIO card needed):
`repro_invoke_subgraph_split.py` — a 3× `nested_compile_region` block of
`relu(h*3+1)` on `spyre` device, `torch.compile(fullgraph=True)`.

## Why it happens

A `nested_compile_region` lowers to an `InvokeSubgraph` HOP: the region body
becomes its own `GraphLowering` (`repeated_subgraph0`) with its own
`torch.fx.Graph`, and the parent graph holds a single `invoke_subgraph` call
node plus a `get_attr` reference to the subgraph module.

During **codegen**, `codegen_subgraph_common` (torch
`codegen/wrapper.py:4185`) recurses into the subgraph correctly under
`V.set_graph_handler(subgraph.graph)` and calls `subgraph.graph.codegen()`.
torch-spyre's `_update_scheduler` monkey-patch (`patches.py:114`) then runs
`CustomPreSchedulingPasses` on the subgraph GraphLowering. So far, correct:
`V.graph` **is** the subgraph.

The bug is in how `split_multi_ops` resolves the FX anchor node it inserts
before. For each splittable `ComputedBuffer` it does:

```python
gl = V.graph                          # = repeated_subgraph0  (correct)
orig_node = next(iter(op.origins))    # line 845
...
with gl.graph.inserting_before(orig_node):   # line 548 / 594
```

For a subgraph buffer, `op.origins` is **a set spanning two fx.Graphs**.
Measured for `repeated_subgraph0_buf0`:

```
op.origins = {
    invoke_subgraph   (call_function InvokeSubgraphHOP)   # PARENT graph
    repeated_subgraph0(get_attr)                          # PARENT graph
    mul               (call_function aten.mul.Tensor)     # SUBGRAPH graph  <-- the real anchor
}
origins whose .graph is gl.graph = [mul]                  # exactly one, unambiguous
```

`next(iter(...))` returns the parent's `invoke_subgraph` node. Its
`.graph` is the parent fx.Graph, not `gl.graph`, so `inserting_before`
asserts `n.graph != self`.

The origin set legitimately spans the HOP boundary (provenance is inherited
across `make_subgraph`); the pass simply must not assume every origin lives
in the current lowering graph.

## Fix sketch

Select the origin node that belongs to the graph being lowered, instead of
an arbitrary one. There is exactly one such node and it is the correct
semantic anchor (the fused body's first compute op), which also carries the
right `meta["val"]` / `stack_trace` that lines 555/596/601–602 copy.

Add a small helper (near the other node helpers in `split_multi_ops.py`):

```python
def _origin_in_graph(origins, g: fx.Graph) -> fx.Node | None:
    """Pick the origin fx.Node that belongs to graph ``g``.

    A buffer lowered inside an invoke_subgraph HOP inherits origins that span
    both the parent graph (the invoke_subgraph call / get_attr nodes) and the
    subgraph's own compute nodes. FX insertion (inserting_before) requires an
    anchor in the *current* graph, so filter to it rather than taking an
    arbitrary origin. Returns None if no origin lives in ``g``.
    """
    return next(
        (n for n in origins if isinstance(n, fx.Node) and n.graph is g),
        None,
    )
```

Use it at the asserting site (lines 841–848):

```python
    # Skip if no FX graph origin node in the current (sub)graph. A buffer
    # lowered inside an invoke_subgraph HOP carries origins from both the
    # parent graph and the subgraph; only the subgraph-local origin is a
    # valid inserting_before anchor here.
    orig_node = _origin_in_graph(op.origins, gl.graph)
    if orig_node is None:
        continue
```

(This replaces both the `if not op.origins` guard, the
`orig_node = next(iter(op.origins))`, and the `isinstance(orig_node, fx.Node)`
guard — `_origin_in_graph` already enforces node-ness and graph membership.)

### Same latent bug, two other sites

Both also do `next(iter(...origins))` and would pick a foreign node for
subgraph buffers:

- **`_get_op_name`, line 698** — used for validation/metadata; picking the
  parent `invoke_subgraph` origin yields the wrong op name (no assert, but
  wrong result). Harden with the same graph-local selection (fall back to the
  old behavior when no graph context / no local origin).
- **`split_multi_ops` env build, line 809** — maps `fx_node -> TensorBox`
  from `gl.name_to_users`. For subgraph buffers the key would be a
  parent-graph node. Only entries actually looked up via `_find_fx_node`
  matter; still worth making graph-consistent to avoid stale keys.

The line-845 fix alone unblocks the assert; 698 and 809 should be fixed in
the same change for correctness.

## Scope note — NOT covered by this fix

`propagate_layouts` and `work_division` emit
`[WARNING] unhandled node type InvokeSubgraph` while walking the parent
graph's operations. In the minimal repro, compilation completes once the
split-insertion is fixed, so these warnings appear benign for the pointwise
case — but they mean neither pass currently propagates layout / work-division
metadata *through* an `InvokeSubgraph` node. That is a separate question from
this assert and should be validated on the real Granite graph (matmuls,
KV-cache mutation) before declaring it cosmetic.

## Blocker 2 (uncovered by fixing Blocker 1): subgraph wrapper codegen

Fixing the split assert let compilation reach subgraph *wrapper* codegen, which
then hit:

```
AttributeError: 'SubgraphPythonWrapperCodegen' object has no attribute
    'generate_const_tensor_fallback'
  torch_spyre/_inductor/ir.py:372  SpyreConstantFallback.codegen
```

Root cause: `SpyrePythonWrapperCodegen.create(is_subgraph=True)` returned the
**stock** `SubgraphPythonWrapperCodegen`, which has none of the Spyre wrapper
overrides (`generate_const_tensor_fallback`, `make_buffer_allocation`,
HBM-pool `generate`, `make_buffer_reuse`, `codegen_free_buffer`,
`allocate_hbm_pool`). The moment a subgraph emitted a Spyre buffer (the
`SpyreConstantFallback` that split_multi_ops materializes) it called a method
that isn't there. Pre-existing gap; the split fix merely first reached it.

Fix (`wrapper.py`): factored the graph-role-agnostic Spyre wrapper behavior
into `_SpyreWrapperCodegenMixin(PythonWrapperCodegen)`, and added
`SpyreSubgraphPythonWrapperCodegen(_SpyreWrapperCodegenMixin,
SubgraphPythonWrapperCodegen)` returned from `create(is_subgraph=True)`.
Role-specific behavior stays on the concrete classes: the top-level wrapper
keeps its import-emitting `write_header`; the subgraph keeps the stock
`write_header = pass` (so imports are not re-emitted) and the stock launcher /
signature plumbing. Sizevars patching (`_patch_sizevars`) runs in both
`__init__`s so the subgraph's own GraphLowering is patched too.

## Verification (DONE — compile-time, host, no card)

1. `python repro_invoke_subgraph_split.py` → **`COMPILE OK, out.shape=(2,64)`**
   (both blockers cleared).
2. `tests/inductor/test_dedup_constants.py`,
   `test_provenance_integration.py` (incl.
   `test_split_multi_ops_records_history_during_real_compile`),
   `test_provenance.py`, `test_propagate_named_dims.py`, `test_log_passes.py`
   → **all pass** (non-subgraph split path unregressed).
3. `pre-commit run --files split_multi_ops.py wrapper.py` → ruff / ruff-format /
   mypy **pass**.

## Blocker 3 (DEVICE-VERIFIED — the layout warnings are NOT cosmetic)

Ran the device gate on a real card:
`whole_forward_e2e.py::test_whole_forward_token_compare_spyre[...granite-3.3-2b-instruct]`.
Blockers 1 & 2 held (compile advanced far into subgraph codegen), then FAILED:

```
InductorError: Unsupported: Spyre backend does not support:
  batchmatmul: cannot restickify any input layout of y to carry y_var=d1
  propagate_layouts.py:804  find_stick_compatible_input_layout
  propagate_layouts.py:846  _matmul_layouts  (y input, generated_var)
  propagate_layouts.py:1826 compute_layouts
  propagate_layouts.py:1826 propagate_spyre_tensor_layouts   ← running on the SUBGRAPH
  (via codegen_subgraph_common → _update_scheduler)
```

### CORRECTED ROOT CAUSE (device-proven 2026-08-11; earlier "layout not propagated across HOP / subgraph inputs starve" is REFUTED)

Instrumented `find_stick_compatible_input_layout` (fscil) and the graph-input
conversion loop with an `SPYRE_DBG_HOP` gate and ran BOTH the whole-forward
device gate and the eager token-compare on the real card. The instrumentation
has since been fully reverted. Evidence (`/tmp/wf_fscil.log`, `/tmp/eager_fscil.log`):

1. **The subgraph input is NOT starved and HAS a real device layout.** The
   input-conversion loop printed:
   `input 'arg2_1' real_input_type=Tensor stl=SpyreTensorLayout(device_size=[64,2,2,1,1,64], stride_map=[256,128,64,64,-1,1])`.
   So `V.get_real_inputs()` for the subgraph returned a REAL device tensor (not a
   fake), `device_tensor_layout()` was not `None`, and the failing matmul reports
   `y.layouts_n=1` (populated). The starvation mechanism does not occur.

2. **The earlier "identical eager vs subgraph `_matmul_layouts` line" was a
   buffer-name collision.** `arg2_1` is a `2048×2048` weight in the eager graph
   but `selected_freqs` in the subgraph — same name, different tensor.

3. **Eager has NO RoPE matmul on `selected_freqs` at all; the subgraph does.**
   Every eager `fscil label=y target_var=d1` is a QKV/MLP weight
   (`host_layout.size=[N,2048]`, `device_size=[N/64,2048,64]`, stick carries d1).
   No eager fscil line has a `[.,2,2,.]` (selected_freqs-shaped) operand. In the
   subgraph, the RoPE contraction becomes `batchmatmul auto_functionalized_subgraph_0_buf11`
   with `y=selected_freqs`, `reduction_var=d2`, `generated_var=d1`.

So the HOP boundary **changes the lowering of the RoPE reduction**, it does not
strip a layout. `apply_rope_matmul` (hf_common.py:298-317) does
`sf.mul(x_.unsqueeze(-3)).sum(4, keepdim=True)` — a mul over a **size-2** axis
followed by `.sum(4)`. In the flat eager graph this stays a pointwise-mul +
reduction. Inside the separately-functionalized `auto_functionalized_subgraph_0`
body, the mul+sum is pattern-matched/decomposed into a `batchmatmul`. That
batchmatmul then requires `y=selected_freqs` to be stickied on
`generated_var=d1` — the **size-2 rotation axis**. `selected_freqs`' only device
layout `[64,2,2,1,1,64]` stickies the innermost `D/2=64` axis (`Mod(d2,64)`);
`d1` maps to the outermost device dim `8*d1+floor(d2/256)` (size 2), which cannot
be restickified onto a 64-element stick. Hence the abort. **The layout is
correct and matches eager's freqs cache — there is nothing to propagate.**

Consequence for the fix: the approved forward/reverse STL-seeding design was
built against the refuted starvation model and does **not** address this. The
real fix must be one of (to decide with the user):
  (a) stop the mul+sum→batchmatmul lowering inside subgraph bodies (force it to
      remain a reduction, matching eager);
  (b) rewrite `apply_rope_matmul` so the contraction is over a stick-friendly
      (≥64) axis instead of the size-2 rotation axis;
  (c) keep RoPE OUT of the region body (apply the freqs matmul before/after the
      `invoke_subgraph`, so the size-2 contraction never enters the HOP).

`work_division`'s `unhandled node type InvokeSubgraph` warning is a separate,
not-yet-reached concern; do not pre-fix it.

## Status summary

| # | Blocker | State |
|---|---------|-------|
| 1 | split_multi_ops `inserting_before` foreign-origin assert | FIXED (`_origin_in_graph`, 3 sites) |
| 2 | subgraph wrapper missing Spyre codegen methods | FIXED (`SpyreSubgraphPythonWrapperCodegen` mixin) |
| 3 | RoPE `.sum(4)` (size-2 contraction) mis-lowers to a `batchmatmul` INSIDE the HOP → freqs must stick on size-2 axis d1 → abort | OPEN — device-PROVEN; NOT a layout-prop gap (starvation refuted); fix = change lowering / rewrite RoPE / hoist RoPE out of region |

Fixes 1 & 2 are correct and self-contained (repro `COMPILE OK`, split-path
tests pass, lint/mypy clean) and are worth landing on their own — they
unblock the pointwise/constant subgraph path and are prerequisites for #3.
```
