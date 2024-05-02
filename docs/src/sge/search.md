# Branch and bound search
Here you find the main function of the package, which is an actual solver.

```@docs
low_energy_spectrum
merge_branches
merge_branches_blur
```
Results of the branch and bound search are stored in a Solution structure.
```@docs
Solution
```

# Droplet search
`SpinGlassPEPS.jl` offers the possibility not only finding low lying energy states, but also droplet excitations. In order to search for droplets, one need to choose the option `SingleLayerDroplets` in `merge_branches`.
```@docs
SingleLayerDroplets
```