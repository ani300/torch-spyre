# RFC: Typed Operation Contracts for the Spyre Compiler Pipeline

**Status:** Draft  
**Authors:** Antoni Viros i Martin  
**Date:** 2026-08-28  
**Related:** PR #3981 (five fix families), PR #4108 (Fix 2 extraction)

---

## Problem Statement

The Spyre compiler pipeline inherits Inductor's representation: flat sympy
index expressions over loop variables, memory dependencies identified by
buffer name, and scheduling driven by data-flow without semantic annotations.
This works when every operation has a unique physical signature. Grouped-query
attention (GQA) breaks that assumption — multiple distinct logical
configurations produce identical physical layouts:

- A window base and a per-tile stride both appear as numeric terms in an index
- A flattened H\*D axis produces two Mod expressions indistinguishable from two
  independent wrapped dimensions
- A shared rank-2 weight and an expanded batched weight have the same physical
  shape at B=1
- Two reads of the same buffer at the same index become one deduplicated
  MemoryDep

Today's pipeline reconstructs logical meaning heuristically at each pass
boundary. PR #3981 demonstrates that this requires five independent fix
families — each compensating for one class of information lost between passes.
These fixes work, but they're fragile: every new pattern that shares a physical
signature with an existing one may need a new heuristic.

### The five information-loss classes

| # | What's lost | Where it's lost | Current fix |
|---|---|---|---|
| 1 | Window base vs. tile stride | Index expression folding | Subtract zero-symbol evaluation in `_tile_advance_expr_from_dep` |
| 2 | Multi-digit axis structure | View/reshape into flat loop | `_mixed_radix_digits` + `_canonicalize_factorized_matmul_dim` |
| 3 | Operand roles and batch ownership | Lowering to untyped deps | `MATMUL_OPERANDS_INFO_KEY` on OpSpec |
| 4 | Reshape sandwich validity | FX-pass pattern matching | Contract-checked rewrite guards |
| 5 | Semantic operand position | Inductor dep deduplication | Occurrence-based edge costing and redirection |

## Design Principles

1. **Information captured once, at the point of maximum knowledge.** Lowering
   knows the logical contract; no later pass should re-derive it.
2. **Contracts are metadata, not a parallel IR.** They annotate existing
   Inductor structures (OpSpec, MemoryDep, ComputedBuffer) rather than
   replacing them.
3. **Incremental adoption.** Each pass can begin reading contracts
   independently. Passes that haven't been updated continue working as today.
4. **Fail-open with diagnostics.** A missing contract means "use heuristic
   fallback" (today's behavior), not "crash." A present contract means "trust
   this over inference."

## Proposed Architecture

### 1. The Operation Contract type

A typed, immutable dataclass attached to every reduction operation at lowering
time:

```python
@dataclass(frozen=True)
class OperationContract:
    """Logical contract for a fused/compiled operation."""

    # What kind of operation this is
    op_kind: Literal["matmul", "sdpa", "conv2d", "pointwise", "reduction"]

    # Ordered logical operand descriptors (positional — index = semantic role)
    operands: tuple[OperandDescriptor, ...]

    # The operation's logical output descriptor
    output: OperandDescriptor

    # For matmul-family: which operand dimensions are M, N, K, Batch
    role_assignment: RoleAssignment | None = None
```

```python
@dataclass(frozen=True)
class OperandDescriptor:
    """One semantic input or output of an operation."""

    # Logical shape as seen by the operation (not the physical buffer shape)
    logical_shape: tuple[int | sympy.Expr, ...]

    # Which dimensions are batch (True) vs. matrix (False)
    # For a shared weight: all batch dims are False
    batch_ownership: tuple[bool, ...] | None = None

    # Whether this operand is shared (stride-zero on batch) vs. owned
    is_shared: bool = False

    # For views into a larger buffer: the constant base offset
    window_base: sympy.Expr | None = None
```

```python
@dataclass(frozen=True)
class RoleAssignment:
    """M/N/K/Batch role mapping for matmul-family operations."""

    # Per-operand, which logical dim index has which role
    # e.g., for [B, M, K] @ [B, K, N] -> [B, M, N]:
    #   lhs_roles = {0: "batch", 1: "M", 2: "K"}
    #   rhs_roles = {0: "batch", 1: "K", 2: "N"}
    #   out_roles = {0: "batch", 1: "M", 2: "N"}
    lhs_roles: dict[int, str]
    rhs_roles: dict[int, str]
    out_roles: dict[int, str]

    # Factorized dimensions: maps a physical dim index to its digit chain
    # e.g., H*D flattening: {1: [("H", 32), ("D", 128)]}
    factorized_dims: dict[int, list[tuple[str, int]]] | None = None
```

### 2. Where contracts are created

**Lowering (single point of truth):**

- `torch_spyre/_inductor/lowering.py`: the existing `_matmul_operands_info`
  becomes a builder for `OperationContract`. Every `lower_mm`, `lower_bmm`,
  and `lower_batched_matmul` call constructs the contract with full knowledge
  of operand shapes, roles, and ownership.

- The SDPA decomposition (`torch_spyre/_inductor/decompositions.py`) annotates
  each internal matmul with window base offsets (the KV block start) and the
  logical tile structure.

- FX-pass rewrites (`temp_passes.py`) that transform matmul structure MUST
  produce a new contract or reject. `_unflatten_mm_to_bmm` validates the
  reshape sandwich and emits a contract with the restored batch axis;
  `_unexpand_shared_rhs_bmm` emits a contract marking the RHS as shared.

**Propagation key:** The contract attaches to the `ComputedBuffer` (via
`OpSpec.op_info`) and travels with it through scheduling and tiling unchanged.
Passes that split or clone a buffer copy the contract to all fragments.

### 3. How each pass consumes contracts

#### Coarse tiling (Fix 1)

**Today:** `_tile_advance_expr_from_dep` subtracts `index.subs({all: 0})` to
strip the window base from the advance expression.

**With contracts:** The `OperandDescriptor.window_base` field explicitly marks
which terms in the index are positional (not per-tile). The advance function
becomes:

```python
def _tile_advance_expr_from_dep(dep, subs, contract_operand=None):
    if contract_operand and contract_operand.window_base is not None:
        index = dep.index - contract_operand.window_base
    else:
        # Fallback: today's heuristic
        index = dep.index
        base = index.subs({s: 0 for s in index.free_symbols})
        index = index - base
    return sympy.expand(index.subs(subs))
```

The heuristic fallback remains for ops without contracts (pointwise, etc.)
where the zero-substitution is always correct.

#### Coordinate recovery (Fix 2)

**Today:** `find_repeat_vars` raises `Unsupported` on multi-Mod patterns
unless `_mixed_radix_digits` recognizes an exact digit chain.

**With contracts:** When `RoleAssignment.factorized_dims` is present,
`compute_coordinates` receives the digit structure directly instead of
reverse-engineering it from the index expression. The flow becomes:

```
lowering sets: factorized_dims = {1: [("H", 32), ("D", 128)]}
    → OpSpec.op_info carries it
    → device_coordinates reads it
    → passes to compute_coordinates as explicit repeat_info
```

`_mixed_radix_digits` remains as validation (assert the index actually matches
the declared structure) rather than discovery.

#### Role assignment and codegen (Fix 3)

**Today:** `_align_matmul_dim_labels` and `_matmul_role_shapes` reconstruct
M/N/K/Batch roles from `operand_shapes` and `batch_dim_owners` stored in
`op_info[MATMUL_OPERANDS_INFO_KEY]`.

**With contracts:** These functions read directly from
`contract.role_assignment`. The reconstruction logic becomes a one-time
migration shim for ops that still use the old dict format:

```python
def get_role_assignment(op_spec: OpSpec) -> RoleAssignment:
    contract = op_spec.op_info.get("operation_contract")
    if contract and contract.role_assignment:
        return contract.role_assignment
    # Fallback: reconstruct from legacy MATMUL_OPERANDS_INFO_KEY
    return _legacy_reconstruct_roles(op_spec)
```

#### FX-pass rewrite validation (Fix 4)

**Today:** Each rewrite in `temp_passes.py` manually validates shapes, ranks,
strides, and ancestry before transforming.

**With contracts:** Rewrites become contract-to-contract transformations. A
rewrite function receives the source contract (attached to the matched node)
and must produce a valid target contract or abort:

```python
def _unflatten_mm_to_bmm(match, ...):
    source_contract = get_contract(match.nodes[-1])
    # ... structural validation ...
    target_contract = OperationContract(
        op_kind="matmul",
        operands=(lhs_desc, rhs_desc),  # rhs is_shared=True, rank-2
        output=out_desc,
        role_assignment=...,
    )
    # Attach to new node
    set_contract(bmm_node, target_contract)
```

The validation logic doesn't disappear — it's still needed to prove the
rewrite is valid — but the **result** is now machine-checkable downstream
rather than trusting that the rewrite preserved an implicit invariant.

#### Restickification edge identity (Fix 5)

**Today:** `finalize_layouts` builds edges from `get_read_writes().reads`,
which Inductor may deduplicate. The branch's `InputEdgeSwapHandler` uses an
occurrence counter to disambiguate.

**With contracts:** The `OperationContract.operands` tuple has explicit
positional identity. When building the restickification edge list:

```python
def build_semantic_edges(op, contract):
    edges = []
    for i, operand_desc in enumerate(contract.operands):
        dep = find_dep_for_operand(op, i, operand_desc)
        edges.append(EdgeCostMap(
            dep, ...,
            semantic_position=i,
            exact_target=operand_desc.is_shared,
        ))
    return edges
```

The edge list is driven by the contract's operand count and order — not by
what Inductor happens to deduplicate. `InputEdgeSwapHandler`'s occurrence
counter becomes unnecessary because each edge already knows its position.

### 4. Contract lifecycle

```
FX graph (post-AOT)
  │
  ▼  temp_passes.py: rewrites attach/transform contracts
Rewritten FX graph
  │
  ▼  lowering.py: creates contracts for all reduction ops
Scheduled IR (OpSpec carries contract in op_info)
  │
  ▼  coarse_tile.py: reads window_base from contract operands
  │  insert_restickify.py: builds edges from contract.operands
  │  optimize_restickify.py: costs edges with contract role info
  │
  ▼  spyre_kernel.py: reads role_assignment for dim labels
  │  superdsc.py: uses role_assignment for SDSC generation
  │
  ▼  Compiled kernel
```

At no point does a pass need to reverse-engineer information that was available
at an earlier stage.

## Migration Strategy

The design is adopted incrementally — one fix family at a time, each a
separate PR:

### Phase 1: Define the types and attach to matmul (weeks 1–2)

- Define `OperationContract`, `OperandDescriptor`, `RoleAssignment` in a new
  `torch_spyre/_inductor/operation_contract.py`
- Migrate `MATMUL_OPERANDS_INFO_KEY` creation in `lowering.py` to produce an
  `OperationContract`
- Add a compatibility shim: `op_info["operation_contract"]` AND the legacy
  `op_info[MATMUL_OPERANDS_INFO_KEY]` are both set (dual-write)
- Consumers (`superdsc.py`, `spyre_kernel.py`) prefer the new contract but
  fall back to legacy dict
- **Validates:** Fix 3 continues working; contract round-trips through OpSpec

### Phase 2: Contract-checked rewrites (weeks 2–3)

- `temp_passes.py` rewrites read the source contract and produce a target
  contract
- Invalid rewrites (where the contract can't be proven valid) are rejected
  rather than silently producing wrong metadata
- **Subsumes:** Fix 4's manual shape/stride/ancestry validation moves into
  contract construction; the rewrite either produces a valid contract or aborts

### Phase 3: Positional edge identity (weeks 3–4)

- `finalize_layouts` and `greedy_local_min_cost` build their edge list from
  `contract.operands` instead of from `get_read_writes().reads`
- Each edge carries its semantic position; deduplication doesn't matter
- `InputEdgeSwapHandler`'s occurrence counter becomes dead code
- **Subsumes:** Fix 5's per-edge costing and occurrence redirection

### Phase 4: Window base annotation (weeks 4–5)

- SDPA decomposition annotates each KV-block matmul's key/value operands with
  `window_base = blk * kv_block_size`
- `_tile_advance_expr_from_dep` reads the annotation directly
- The zero-substitution heuristic remains as a fallback for non-annotated ops
- **Subsumes:** Fix 1's base-offset subtraction for annotated ops

### Phase 5: Factorized dimension metadata (weeks 5–6)

- Lowering annotates H\*D flattening with `factorized_dims`
- `compute_coordinates` receives the digit structure from the contract rather
  than discovering it via `_mixed_radix_digits`
- `_mixed_radix_digits` becomes an assertion (validate that the index matches
  the declared structure) rather than the discovery mechanism
- **Subsumes:** Fix 2's reverse-engineering of digit chains

## What This Does NOT Change

- **Inductor's IR.** We annotate, not replace. MemoryDep, ComputedBuffer, and
  LoopLevel IR remain unchanged.
- **Upstream torch compatibility.** The contract is Spyre-specific metadata in
  `op_info`; upstream Inductor never sees it.
- **Non-reduction ops.** Pointwise, elementwise, and simple reductions don't
  need contracts — their physical signatures are unambiguous.
- **The heuristic fallbacks.** Every consumer retains a "no contract present"
  code path that does exactly what today's code does. Contracts add
  information; they don't gate compilation.

## Success Criteria

1. All five fix families' regression tests pass with contracts enabled
2. No regression test requires the heuristic fallback path for a matmul/SDPA op
3. Adding a new attention pattern (e.g., sliding window, cross-attention)
   requires only a new contract at lowering time — no new heuristic in any
   downstream pass
4. The `_mixed_radix_digits`, `_tile_advance_expr_from_dep` base subtraction,
   and `InputEdgeSwapHandler` occurrence counter are deletable (behind a flag)
   once all annotated ops have contracts

## Open Questions

1. **Should contracts be frozen after lowering, or can passes refine them?**
   For example, coarse tiling splits an operation into tiles — should each tile
   get a sub-contract? Current proposal: frozen after creation; passes read but
   never write.

2. **Should non-matmul reductions (softmax, layernorm) get contracts?** They
   don't need role assignment, but they could benefit from window-base
   annotation if they participate in tiled SDPA.

3. **Contract identity and kernel caching.** Today OpSpec identity drives
   kernel cache hits. Adding a contract changes the identity space. Should
   contract fields contribute to cache keys, or should only the physical IR
   matter for caching?

4. **Relationship to upstream LoopLevel IR evolution.** If upstream Inductor
   adds its own semantic annotations (there are proposals), should we align
   our contract schema with theirs?
