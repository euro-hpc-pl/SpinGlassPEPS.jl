# Introduction
A [Julia](http://julialang.org) package for finding low energy spectrum of Ising spin systems. Part of [SpinGlassPEPS](https://github.com/euro-hpc-pl/SpinGlassPEPS.jl) package.

This part of the documentation is dedicated to describing the `SpinGlassEngine.jl` package, which serves as the actual solver. First, we will demonstrate how to construct a tensor network using the clustered Hamiltonian obtained with the `SpinGlassNetworks.jl` package. Next, we discuss the parameters necessary for conducting calculations, which the user should provide. Finally, we present functions that enable the discovery of low-energy spectra.

