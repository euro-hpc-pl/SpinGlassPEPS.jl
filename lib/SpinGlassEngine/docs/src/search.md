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
`SpinGlassPEPS.jl` provides the capability to find not only low-energy states but also droplet excitations. To search for droplets, the `SingleLayerDroplets` option must be selected in the `merge_branches` function.

Droplets are identified during the optional `merge_branches` step, which can be invoked within the `low_energy_spectrum` function that runs the branch-and-bound algorithm. This search focuses on finding diverse excitations within a specific energy range above the ground state. An excitation is accepted only if its Hamming distance from any previously identified excitation exceeds a predefined threshold.

This behavior is controlled by two key parameters:
* `energy_cutoff`: Defines the maximum allowed energy above the ground state for considering an excitation.
* `hamming_cutoff`: Sets the minimum Hamming distance required between excitations for them to be classified as distinct.
By adjusting these parameters, users can search for different excitations while ensuring that only sufficiently distinct ones are included.
```@docs
SingleLayerDroplets
```