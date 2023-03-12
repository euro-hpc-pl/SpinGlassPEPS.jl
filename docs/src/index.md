```@meta
Author = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams, Anna Dziubyna, Tomasz Śmierzchalski"
```

# Welcome to SpinGlassPEPS documentation!

`SpinGlassPEPS` is an open-source Julia package for numerical computation in quantum information theory. 

!!! info "Star us on GitHub!" 
    If you have found this library useful, please consider starring the GitHub repository. This gives us an accurate lower bound of the satisfied users.


## Overview
In this section we will provide a condensed overview of the package.

`SpinGlassPEPS.jl` is a collection of Julia packages bundled together under a single package `SpinGlassPEPS`. It can be installed using the Julia package manager for Julia v1.6 and higher. Inside the Julia REPL, type ] to enter the Pkg REPL mode and then run
```julia
using Pkg; 
Pkg.add("SpinGlassPEPS")
```
The package `SpinGlassPEPS` includes:
* `SpinGlassTensors.jl` - contains functions used in tensor network contractions
* `SpinGlassNetworks.jl` - creates factor graph and Ising spin-glass model
* `SpinGlassEngine.jl` - search for low energy spectrum using PEPS tensor network




## Our goals

`SpinGlassPEPS.jl` was created to heuristically solve Ising-type optimization problems defined on quasi-2d lattices.
It enables to compute conditional probabilities and find the most probable states according to Gibbs distribution by contracting tensor networks. It is a powerful tool to reconstruct the low-energy spectrum of the model. 

We aim to provide fast, reliable and easy to use emulator of D-Wave ``2000``Q quantum annealers. Our solver calculates ``L \ll 2 ^N`` low energy states (and their corresponding energies) for ``N \le 2048``. 

## Citing SpinGlassPEPS
If you use `SpinGlassPEPS` for academic research and wish to cite it, please use the following paper:

K. Jałowiecki, K. Domino, A. M. Dziubyna, M. M. Rams, B. Gardas and Ł. Pawela, *“SpinGlassPEPS.jl: software to emulate quantum annealing processors”*

## Contributing
Contributions are always welcome:
* Please report any issues and bugs that you encounter in Issues
* Questions about `SpinGlassPEPS.jl` can be asked by directly opening up an Issue on its GitHub page
* If you plan to contribute new features, extensions, bug fixes, etc, please first open an issue and discuss the feature with us.

!!! info "Report the bug" 
    Filling an issue to report a bug, counterintuitive behavior, or even to request a feature is extremely valuable in helping us prioritize what to work on, so don't hestitate.

