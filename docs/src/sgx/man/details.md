# GPU kernels

The high-level `exhaustive_search` and `exhaustive_search_bucket` functions are preferred
for normal use. The lower-level kernels are exposed for specialized CUDA workflows.

```julia
using SpinGlassPEPS
using CUDA

N = 8
graph = generate_random_graph(N)
device_graph = CuArray(graph)
energies = CUDA.zeros(Float32, 2^N)

threads = 512
blocks = cld(length(energies), threads)
CUDA.@cuda threads = threads blocks = blocks kernel(device_graph, energies)

state_codes = sortperm(Array(energies)) .- 1
```

`kernel_qubo` performs the analogous calculation for a QUBO matrix. Kernel array indices
are one-based, while the encoded state written for a result is the corresponding
zero-based integer.
