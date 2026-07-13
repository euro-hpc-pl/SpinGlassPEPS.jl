# Quickstart

Exhaustive-search functionality is included in `SpinGlassPEPS.jl`; no component package
needs to be installed separately.

```julia
using Pkg
Pkg.add("SpinGlassPEPS")

using SpinGlassPEPS
```

CUDA routines are available when a supported GPU and functional CUDA installation are
present. The CPU `brute_force` method remains available without a GPU.
