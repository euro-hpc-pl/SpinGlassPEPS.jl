# SpinGlassPEPS.jl 

Welcome to `SpinGlassPEPS.jl`, a open-source Julia package designed for heuristically finding low-energy configurations of generalized Potts models, including Ising and QUBO (Quadratic Unconstrained Binary Optimization) problems. 

| **Documentation** | **Digital Object Identifier** |
|:-----------------:|:-----------------------------:|
|[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://euro-hpc-pl.github.io/SpinGlassPEPS.jl/dev/)| TO DO |



* # Our goals

`SpinGlassPEPS.jl` was created to heuristically solve Ising-type optimization problems defined on quasi-2D lattices or Random Markov Fields (RMF) on 2D rectangular lattices. This package combines advanced heuristics to address optimization challenges and employs tensor network contractions to compute conditional probabilities to identify the most probable states according to the Gibbs distribution. `SpinGlassPEPS.jl` is a tool for reconstructing the low-energy spectrum of Ising spin glass Hamiltonians and RMF Hamiltonians. Beyond energy computations, the package offers insights into spin configurations, associated probabilities, and retains the largest discarded probability during the branch and bound optimization procedure. Notably, `SpinGlassPEPS.jl` goes beyond ground states, introducing a unique feature for identifying and analyzing spin glass droplets — collective excitations crucial for understanding system dynamics beyond the fundamental ground state configurations.


## Package architecture
The package `SpinGlassPEPS.jl` includes:

* `SpinGlassTensors.jl` - Package containing  essential tools for creating and manipulating tensors that constitute the PEPS network, with support for both CPU and GPU utilization. It manages core operations on tensor networks, including contraction, using the boundary Matrix Product State approach. This package primarily functions as a backend, and users generally do not interact with it directly.

* `SpinGlassNetworks.jl` - Package  facilitating the generation of an Ising graph from a given instance using a set of standard inputs (e.g., instances compatible with the Ocean environment provided by D-Wave) and suports clustering to create effective Potts Hamiltonians

* `SpinGlassEngine.jl` - The main package , consisting of routines for executing the branch-and-bound method (with the ability to leverage the problem’s locality) for a given Potts instance. It also includes capabilities for reconstructing the low-energy spectrum from identified localized excitations and provides a tensor network constructor.

# Citing
See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).