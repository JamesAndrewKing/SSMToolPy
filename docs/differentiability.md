# Differentiability Status

| Public API | Status | Notes |
| --- | --- | --- |
| `MultiIndexPolynomial` | not differentiable | Frozen data container; evaluation APIs carry differentiability. |
| `nsumk` | not differentiable | Discrete integer tuple enumeration. |
| `sub2multiind` | not differentiable | Discrete one-based subscript counting. |
| `expand_multiindex` | differentiable | Tested with `jax.jit` and `jax.grad`. |
| `expand_multiindex_derivative` | differentiable | Implemented through JAX autodiff; tested indirectly against `jax.jacfwd`. |
| `tensor_to_multi_index` | not differentiable | Discrete grouping/conversion from dense tensor entries. |
| `multi_index_to_tensor` | not differentiable | Discrete canonical tensor placement. |
| `multi_addition` | not differentiable | Discrete index algebra. |
| `multi_subtraction` | not differentiable | Discrete index algebra and filtering. |
| `multi_nsumk` | not differentiable | Discrete multi-index partition enumeration with optional filtering and deduplication. |
| `multi_index_2_ordering` | not differentiable | Discrete combinatorial ordering lookup. |
| `khatri_rao_product` | differentiable | Tested with `jax.jacfwd`. |
| `expand_tensor` | differentiable | Tested with `jax.jit` and `jax.grad`. |
| `expand_tensor_derivative` | differentiable | Tested against `jax.jacfwd`. |
| `frc_ab` | differentiable | Tested with `jax.grad` and `jax.vmap`. |
| `reduced_to_full` | differentiable | For fixed polynomial structure; tested with `jax.jit`. Non-autonomous branch uses fixed Python structure. |
| `reduced_to_full_complex` | differentiable | For fixed polynomial/forcing structure; transform coverage not yet verified. |
