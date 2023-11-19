# Introduction
A clustered Hamiltonian is a graphical representation that allows for a convenient and intuitive way to describe the structure of a network.

The concept of a clustered Hamiltonian within `SpinGlassNetworks.jl` introduces a powerful mechanism for organizing spins into desired geometries, facilitating a structured approach to modeling complex spin systems. Analogous to a factor graph, the clustered Hamiltonian involves nodes that represent tensors within the underlying network. The edges connecting these nodes in the clustered Hamiltonian correspond to the indices shared between the respective tensors in the tensor network.

The Ising problem in translates to:
$$ H(\conf{x}{\Nbar}) = \sum_{\langle m,n\rangle \in \mathcal{F}} E_{x_m x_n} + \sum_{n=1}^{\Nbar} E_{x_n} $$
$\mathcal{F}$ forms a 2D graph, see the figure below, where we indicate nearest-neighbour interactions with blue lines, and diagonal connections with green lines.
Each $x_n$ takes $d$ values with  $d=2^4$ for square diagonal, $d=2^{24}$ for Pegasus and $2^{16}$ for Zephyr geometry. 
$E_{x_n}$ is an intra--node energy of the corresponding binary-variables configuration, and $E_{x_n x_m}$ is inter--node energy.

```@raw html
<img src="../images/clh.pdf" width="200%" class="center"/>
```

```@docs
clustered_hamiltonian
```

## Simple example

```@example
using SpinGlassNetworks

# Prepare simple instance
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"
ig = ising_graph(instance)

# Create clustered Hamiltonian.
cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((5,5,4))
)
```