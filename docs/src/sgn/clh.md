# Introduction
A clustered Hamiltonian is a graphical representation that allows for a convenient and intuitive way to describe the structure of a network.

The concept of a clustered Hamiltonian within `SpinGlassNetworks.jl` introduces a powerful mechanism for organizing spins into desired geometries, facilitating a structured approach to modeling complex spin systems. Analogous to a factor graph, the clustered Hamiltonian involves nodes that represent tensors within the underlying network. The edges connecting these nodes in the clustered Hamiltonian correspond to the indices shared between the respective tensors in the tensor network.

```@docs
clustered_hamiltonian
```

## Simple example

```@example
using SpinGlassNetworks

# Prepare simple instance
instance = Dict((1, 1) => 1.0, (2, 2) => 0.5, (3, 3) => -0.25, 
(1, 2) => -1.0, (2, 3) => 1.0)
ig = ising_graph(instance)

# Create clustered Hamiltonian.
cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((3, 1, 1))
)
```