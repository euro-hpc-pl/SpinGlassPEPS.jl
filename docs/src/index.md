```@meta
Author = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams, Anna Dziubyna"
```

# Welcome to SpinGlassPEPS documentation!
## Home
`SpinGlassPEPS` is an open-source Julia package for numerical computation in quantum information theory. 

!!! info "Star us on GitHub!" 
    If you have found this library useful, please consider starring the GitHub repository. This gives us an accurate lower bound of the satisfied users.

## Getting started
In this section we will provide a condensed overview of the package.

`SpinGlassPEPS.jl` is a collection of Julia packages bundled together under a single package `SpinGlassPEPS`. It can be installed using the Julia package manager for Julia v1.5 and higher. Inside the Julia REPL, type ] to enter the Pkg REPL mode and then run
```julia
using Pkg; 
Pkg.add("SpinGlassPEPS")
```
The package `SpinGlassPEPS` includes:
* `SpinGlassTensors.jl` - contains auxiliary functions used in `SpinGlassPEPS`
* `SpinGlassNetworks.jl` - creates factor graph and Ising spin-glass model
* `SpinGlassEngine.jl` - search for low energy spectrum using PEPS and MPS

## Our goals

`SpinGlassPEPS.jl` was created to heuristically solve Ising-type optimization problems defined on quasi-2d lattices.
It enables to compute conditional probabilities and find the most probable states according to Gibbs distribution by contracting tensor networks. It is a powerful tool to reconstruct the low-energy spectrum of the model. 

We aim to provide fast, reliable and easy to use emulator of D-Wave ``2000``Q quantum annealers. Our solver calculates ``L \ll 2 ^N`` low energy states (and their corresponding energies) for ``N \le 2048``. 

## Citing SpinGlassPEPS
If you use `SpinGlassPEPS` for academic research and wish to cite it, please use the following paper:

K. Jałowiecki, K. Domino, A. M. Dziubyna, M. M. Rams, B. Gardas and Ł. Pawela, *“SpinGlassPEPS.jl: software to emulate quantum annealing processors”*

# Examples
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Introduction
We consider a classical Ising Hamiltonian
```math
E = -\sum_{<i,j> \in \mathcal{E}} J_{ij} s_i s_j - \sum_j h_i s_j.
```
where ``s`` is a configuration of ``N`` classical spins taking values ``s_i = \pm 1``
and ``J_{ij}, h_i \in \mathbb{R}`` are input parameters of a given problem instance. 
Nonzero couplings ``J_ij`` form a graph ``\mathcal{E}``. 

## MPS
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the matrix product states (MPS) are.

## PEPS
## Examples of building documentation from docstrings
```@docs
SpinGlassNetworks.ising_graph
```
```@docs
SpinGlassNetworks.gibbs_tensor
```
```@docs
SpinGlassNetworks.energy
```
```@docs
SpinGlassNetworks.brute_force
```
```@docs
SpinGlassTensors.compress
```

## Finding structure of low energy states
