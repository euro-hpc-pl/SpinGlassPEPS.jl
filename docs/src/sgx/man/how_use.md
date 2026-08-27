# Usage

Construct an Ising graph from a dictionary or an instance file, then choose the CPU or
GPU implementation:

```julia
using SpinGlassPEPS
using CUDA

instance = Dict(
    (1, 1) => -0.2,
    (2, 2) => 0.1,
    (1, 2) => -1.0,
)
ig = ising_graph(instance)

cpu_result = brute_force(ig; num_states = 4)

if CUDA.functional()
    all_gpu_states = exhaustive_search(ig)
    lowest_gpu_states = exhaustive_search_bucket(ig, 4)
end
```

State identifiers returned by the exhaustive GPU routines are zero-based integer state
codes. Use the corresponding energies in the returned spectrum to compare results.
