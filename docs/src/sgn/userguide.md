# Introduction
This section provides examples of how to use the 'ising_graph' function to generate an Ising model graph and the 'factor_graph' function to convert the Ising graph into a factor graph.

# Ising Graph
The Ising model is a mathematical model used to describe the behavior of interacting particles, such as atoms or molecules, in a magnetic field. In the Ising model, each particle is represented as a binary variable $s_i$ that can take on the values of either +1 or -1. The total energy of the system is given by the Hamiltonian:

$$H =  \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

where $J_{ij}$ is the coupling constant between particles $i$ and $j$, $h_i$ is the external magnetic field at particle $i$, and the sum is taken over all pairs of particles and all particles in the system $\mathcal{E}$, respectively.


## simple examle


## Generalisation

The 'ising_graph' function is quite general function. One can easly convert instance into another hamiltonian convention, ex.
```math
H = - \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j - \mu \sum_{i} h_i s_i
```
This conversion is directed by `sgn` parameter of `ising_graph`. It is wort doing, because then all `energy` functions will work as one intended.


'ising_graph' also allows for building other spin system, not only spin-``\frac{1}{2}`` system. For example one may build chain where sites allows for 3 spin values.



# factor graph

A factor graph is a graphical representation that allows for a convenient and intuitive way to describe the structure of a tensor network.

A factor graph consists of two types of nodes: variable nodes and factor nodes. Variable nodes represent the tensors in the tensor network, and factor nodes represent the operations that are applied to those tensors.

Each variable node in the factor graph corresponds to a tensor in the tensor network, and each factor node corresponds to a tensor contraction or operation that is applied to one or more of those tensors. The edges between the nodes in the factor graph represent the indices that are shared between the corresponding tensors in the tensor network.

## Simple example



## Chimera graphs
The Chimera graph is a type of graph architecture used in quantum computing systems, particularly in the quantum annealing machines developed by D-Wave Systems. It is a two-dimensional lattice of unit cells, each consisting of a bipartite graph of $K_{4,4}$ complete bipartite subgraphs. Futer details can be found [here](https://docs.dwavesys.com/docs/latest/c_gs_4.html#chimera-graph).

