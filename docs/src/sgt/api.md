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
