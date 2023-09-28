# Introduction
This section provides examples of how to use the 'ising_graph' function to generate an Ising model graph and the 'clustered_hamiltonian' function to convert the Ising graph into a clustered Hamiltonian.

# Ising Graph
The Ising model is a mathematical model used to describe the behavior of interacting particles, such as atoms or molecules, in a magnetic field. In the Ising model, each particle is represented as a binary variable $s_i$ that can take on the values of either +1 or -1. The total energy of the system is given by the Hamiltonian:

$$H =  \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

where $J_{ij}$ is the coupling constant between particles $i$ and $j$, $h_i$ is the external magnetic field at particle $i$, and the sum is taken over all pairs of particles and all particles in the system $\mathcal{E}$, respectively.


## Simple example
```@example
using SpinGlassNetworks
# Create Ising instance
instance = Dict((1, 1) => 1.0, (2, 2) => 0.5, (3, 3) => -0.25, 
(1, 2) => -1.0, (2, 3) => 1.0)

# Generate Ising graph
ig = ising_graph(instance)

# View graph properties
@show biases(ig), couplings(ig)
```

## Generalisation

The 'ising_graph' function is quite general function. One can easly convert instance into another hamiltonian convention, ex.
```math
H = - \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j - \mu \sum_{i} h_i s_i
```

'ising_graph' also allows for building other spin system, not only spin-``\frac{1}{2}`` system. For example one may build chain where sites allows for 3 spin values.

```@example
using SpinGlassNetworks

# Create spin chain instance 
instance = Dict((1, 1) => 1.0, (2, 2) => 0.5, (3, 3) => -0.25, 
(1, 2) => -1.0, (2, 3) => 1.0)

rank_override = Dict(1 => 3, 2 => 3, 3 => 3)

sg = ising_graph(instance, rank_override=rank_override)

# see ranks of each site
rank_vec(sg)
```

# Clustered Hamiltonian

A clustered Hamiltonian is a graphical representation that allows for a convenient and intuitive way to describe the structure of a tensor network.

A clustered Hamiltonian consists of two types of nodes: variable nodes and factor nodes. Variable nodes represent the tensors in the tensor network, and factor nodes represent the operations that are applied to those tensors.

Each variable node in the clustered Hamiltonian corresponds to a tensor in the tensor network, and each factor node corresponds to a tensor contraction or operation that is applied to one or more of those tensors. The edges between the nodes in the clustered Hamiltonian represent the indices that are shared between the corresponding tensors in the tensor network.

## Simple example

```@example
using SpinGlassNetworks

# Prepare simple instance
instance = Dict((1, 1) => 1.0, (2, 2) => 0.5, (3, 3) => -0.25, 
(1, 2) => -1.0, (2, 3) => 1.0)
ig = ising_graph(instance)

# Create factor graph.
cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((3, 1, 1))
)
```

## Pegasus graphs
The Pegasus graph is a type of graph architecture used in quantum computing systems, particularly in the quantum annealing machines developed by D-Wave Systems. It is a two-dimensional lattice of unit cells, each consisting of a bipartite graph of $K_{4,4}$ complete bipartite subgraphs. Futer details can be found [here](https://docs.dwavesys.com/docs/latest/c_gs_4.html#pegasus-graph).
```@raw html
<img src="../images/peg.pdf" width="200%" class="center"/>
```

```@example
using SpinGlassEngine, SpinGlassNetworks, LabelledGraphs

# load Chimera instance and create Ising graph
instance = "$(@__DIR__)/../../src/instances/pegasus_random/P4/RAU/001_sg.txt"
ig = ising_graph(instance)

# Loaded instance is pegasus graph
m = 3
n = 3
t = 3

cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = pegasus_lattice((m, n, t))
)

println("Number of nodes in oryginal instance: ", length(LabelledGraphs.vertices(ig)), "\n", " Number of nodes in factor graph: ", length(LabelledGraphs.vertices(cl_h)))
```


## Zephyr graphs
The Zephyr graph is a type of graph architecture used in quantum computing systems, particularly in the quantum annealing machines developed by D-Wave Systems. Futer details can be found [here](https://docs.dwavesys.com/docs/latest/c_gs_4.html#zephyr-graph).
```@raw html
<img src="../images/zep.pdf" width="200%" class="center"/>
```
```@example
using SpinGlassEngine, SpinGlassNetworks, LabelledGraphs

# load Chimera instance and create Ising graph
instance = "$(@__DIR__)/../../src/instances/zephyr_random/Z3/RAU/001_sg.txt"
ig = ising_graph(instance)

# Loaded instance is zephyr graph
m = 6
n = 6
t = 4

cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = zephyr_lattice((m, n, t))
)

println("Number of nodes in oryginal instance: ", length(LabelledGraphs.vertices(ig)), "\n", " Number of nodes in factor graph: ", length(LabelledGraphs.vertices(cl_h)))
```