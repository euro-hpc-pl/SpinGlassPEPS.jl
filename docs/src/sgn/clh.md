# Introduction
A Potts Hamiltonian is a graphical representation that allows for a convenient and intuitive way to describe the structure of a network.

The concept of a Potts Hamiltonian within `SpinGlassNetworks.jl` introduces a mechanism for organizing spins into desired clustered geometries, facilitating a structured approach to modeling complex spin systems. 

```@docs
potts_hamiltonian
```

## Simple example

```@example
using SpinGlassNetworks

# Load instance
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"
ig = ising_graph(instance)

# Create Potts Hamiltonian
potts_h = potts_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((5,5,4))
)
```