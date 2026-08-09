# Library

---
```@meta
CurrentModule = SpinGlassPEPS.SpinGlassTensors
```
## Additional methods for `Base` and `LinearAlgebra`
```@docs
left_nbrs_site
right_nbrs_site
project_ket_on_bra
kernel_batch_size
```
## MPS


## Compresions and Contractions
```@docs
update_env_left
update_env_right
update_reduced_env_right
```

## Projectors

```@docs
PoolOfProjectors
get_projector!
add_projector!
Base.empty!(::PoolOfProjectors, ::Symbol)
Base.copy(::PoolOfProjectors)
```

## Task-scoped execution context

Internals backing the device-memory governor and the truncation-error reporting
described under [Concurrent sweeps and error control](@ref). Both quantities are
dynamically scoped per task so that concurrently running solves see their own
values without threading them through every kernel signature.

```@docs
device_memory_budget
record_truncation!
Base.empty!(::TruncationLog)
DevicePeak
DEVICE_PEAK_PROBE
device_peak_bytes
probe_device_peak!
```
