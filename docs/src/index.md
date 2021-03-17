```@meta
Author = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams, Anna Dziubyna"
```
# Welcome to SpinGlassPEPS documentation!
## Home
SpinGlassPEPS is an open-source Julia package for numerical computation in quantum information theory. 

## Getting started
SpinGlassPEPS.jl is a collection of Julia packages bundled together under a single package SpinGlassPEPS. To install this bundle you can do:

## Our goals

SpinGlassPEPS.jl was created to heuristically solve Ising-type optimization problems defined on quasi-2d lattices.
It enables to compute conditional probabilities and find the most probable states according to Gibbs distribution by contracting tensor networks. It is a powerful tool to reconstruct the low-energy spectrum of the model. 

We aim to provide fast, reliable and easy to use emulator of D-Wave $2000$Q quantum annealers. Our solver calculates $L \ll 2 ^N$ low energy states (and their corresponding energies) for $N \le 2048$. 

## Citation

SpinGlassPEPS is based on the paper K. Jałowiecki, K. Domino, A. M. Dziubyna, M. M. Rams, B. Gardas and Ł. Pawela, “SpinGlassPEPS.jl: software to emulate quantum annealing processors”