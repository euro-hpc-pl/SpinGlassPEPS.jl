# Ising model

The Ising model is a mathematical model used to describe the behavior of interacting particles, such as atoms or molecules, in a magnetic field. In the Ising model, each particle is represented as a binary variable $s_i$ that can take on the values of either +1 or -1. The total energy of the system is given by the Hamiltonian:

$H =  \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i} h_i s_i$

where $J_{ij}$ is the coupling constant between particles $i$ and $j$, $h_i$ is the external magnetic field at particle $i$, and the sum is taken over all pairs of particles and all particles in the system $\mathcal{E}$, respectively.

In `SpinGlassPEPS.jl` package, an Ising graph can be created using the command `ising_graph`.
```@docs
ising_graph
```

## Simple example
In this simple example below we show how to create Ising graph of a instance given as txt file in a format (i, j, Jij). The resulting graph has vertices corresponding to positions of spins in the system and edges defining coupling strength between spins. Each vertex contains information about local field.

```@example
using SpinGlassNetworks
# Create Ising instance
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"
ig = ising_graph(instance)

# View graph properties
@show biases(ig), couplings(ig)
```