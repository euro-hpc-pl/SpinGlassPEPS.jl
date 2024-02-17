# Library

```@meta
CurrentModule = SpinGlassNetworks
```

## Ising Graphs
```@docs
inter_cluster_edges
prune
couplings
```

## Clustered Hamiltonian
```@docs
split_into_clusters
decode_clustered_hamiltonian_state
rank_reveal
energy
energy_2site
cluster_size
bond_energy
exact_cond_prob
truncate_clustered_hamiltonian
```

## Belief propagation
```@docs
local_energy
interaction_energy
get_neighbors
MergedEnergy
update_message
merge_vertices_cl_h
projector
SparseCSC
```

## Projectors
```@docs
PoolOfProjectors
get_projector!
add_projector!
empty!
```

## Spectrum
```@docs
Spectrum
matrix_to_integers
gibbs_tensor
brute_force
```

## Truncate
```@docs
truncate_clustered_hamiltonian_1site_BP
truncate_clustered_hamiltonian_2site_energy
select_numstate_best
```

## Auxiliary Functions
```@docs
zephyr_to_linear
load_openGM
```