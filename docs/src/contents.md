<!-- # Contents

The module `SpinGlassPEPS.jl` re-exports all following functionality.

## SpinGlassTensors
`SpinGLassTensors` is a collection of many auxiliary functions.
* `base.jl` - provides auxiliary functions to check MPS properties
* `compressions.jl` - an interface to truncate, canonise and compress MPS variationally.
```@docs
SpinGlassTensors.compress
```
* `contractions.jl` - offers auxiliary functions to contract the tensor network such as preparing left or right right environment of MPS
* `identities.jl` - 
* `linear_algebra_ext.jl` - wraper to QR and SVD

## SpinGlassNetworks
* `factor.jl` - introduces factor graph
* `ising.jl` - creates the Ising spin glass model
```@docs
SpinGlassNetworks.ising_graph
```
* `lattice.jl` - forms a square lattice
* `spectrum.jl` - is a collection of functions to calculate low energy spectrum. Enables to compute Ising energy and Gibbs state for classical Ising Hamiltonian.
```@docs
SpinGlassNetworks.gibbs_tensor
```
```@docs
SpinGlassTensors.left_env
```
```@docs
SpinGlassNetworks.energy
```
```@docs
SpinGlassNetworks.brute_force
```
* `states.jl`

## SpinGlassEngine
`SpinGlassEngine` is a main module of the package `SpinGlassPEPS` which allows for searching for the low energy spectrum using branch and bound algorithm.
* `MPS_search.jl` - searching for the low energy spectrum on quasi-1d graph
* `PEPS.jl` - introduces PEPS tensor network and contracts it using boundary matrix prduct state approach
* `search.jl` - searching for the low-energy spectrum on a quasi-2d graph -->