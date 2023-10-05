# Library

---

## Search
```@docs
SearchParameters
Solution
empty_solution
low_energy_spectrum
gibbs_sampling
bound_solution
no_merge
merge_branches
branch_energy
SingleLayerDroplets
Flip
Droplet
```

## Core 
```@docs
error_measure
conditional_probability
update_energy
boundary
local_state_for_node
boundary_indices
sweep_gauges!
update_gauges!
Gauges
GaugeInfo
PEPSNode
SuperPEPSNode
```

## Contractor
```@docs
MpoLayers
MpsParameters
MpsContractor
PEPSNetwork
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
LatticeTransformation
rotation
reflection
all_lattice_transformations
vertex_map
check_bounds
```