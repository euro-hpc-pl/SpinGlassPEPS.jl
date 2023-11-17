# Library

---

## Search
```@docs
Solution
empty_solution
gibbs_sampling
bound_solution
no_merge
branch_energy
```

## Core 
```@docs
error_measure
conditional_probability
update_energy
boundary
boundary_indices
Gauges
GaugeInfo
PEPSNode
SuperPEPSNode
```

## Contractor
```@docs
MpoLayers
layout
sparsity
strategy
mpo
mps_top
mps
mps_approx
dressed_mps
right_env
left_env
clear_memoize_cache
clear_memoize_cache_after_row
```

## Operations
```@docs
vertex_map
check_bounds
```

## Droplets
```@docs
SingleLayerDroplets
Flip
Droplet
NoDroplets
hamming_distance
unpack_droplets
perm_droplet
filter_droplets
my_push!
diversity_metric
merge_droplets
flip_state

```

## PEPS
```@docs
SpinGlassEngine.local_energy
SpinGlassEngine.interaction_energy
normalize_probability
initialize_gauges!
decode_state
SpinGlassEngine.bond_energy
SpinGlassEngine.projector
spectrum
is_compatible
ones_like
tensor_map
size
exact_spectrum
discard_probabilities!
mod_wo_zero
exact_marginal_probability
_normalize
projectors_site_tensor
branch_probability
exact_conditional_probability
branch_solution
gauges_list
SquareCrossDoubleNode
SquareSingleNode
branch_energies
_equalize
nodes_search_order_Mps
sampling
VirtualDoubleNode
merge_branches_blur
fuse_projectors
local_spins
tensor
branch_states
precompute_conditional
```