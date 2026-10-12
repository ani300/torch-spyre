# Release notes

## Unreleased

- BF16 collective tensors use their standard DL16 device storage. Eager
  collectives require contiguous tensors with zero storage offset and a
  descriptor covering their own logical extent; unsupported storage views
  now raise an explicit error. Compiled broadcast, allgather, and allreduce
  materialize view inputs automatically, with in-place allreduce copying the
  completed result back to its original view after the wait. See
  [collective tensor requirements](runtime/index.md#bf16-collective-tensors).
