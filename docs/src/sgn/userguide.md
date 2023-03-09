# Introduction
This section provides examples of how to use the 'ising_graph' function to generate an Ising model graph and the 'factor_graph' function to convert the Ising graph into a factor graph.

# Ising Graph
The Ising model is a mathematical model used to describe the behavior of interacting particles, such as atoms or molecules, in a magnetic field. In the Ising model, each particle is represented as a binary variable $s_i$ that can take on the values of either +1 or -1. The total energy of the system is given by the Hamiltonian:

$$H =  \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

where $J_{ij}$ is the coupling constant between particles $i$ and $j$, $h_i$ is the external magnetic field at particle $i$, and the sum is taken over all pairs of particles and all particles in the system $\mathcal{E}$, respectively.


## simple examle
```jldoctest
using SpinGlassNetworks
# Create Ising instance
instance = Dict((1, 1) => 1.0, (2, 2) => 0.5, (3, 3) => -0.25, (4, 4) => 0.0, (1, 2) => -1.0, (2, 3) => 1.0, (3, 1) => -1.5)

# Generate Ising graph
ig = ising_graph(instance)

# View graph properties
@show biases(ig)
@show couplings(ig)

# Remove isolated vertice
ig2 = prune(ig)

# View biases fo smaller graph
@show biases(ig2)

# output
biases(ig) = [1.0, 0.5, -0.25, 0.0]
couplings(ig) = [0.0 -1.0 -1.5 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
biases(ig2) = [1.0, 0.5, -0.25]


```

# factor graph
