## Lattice
Within the `SpinGlassNetworks.jl` package, users have the flexibility to construct various lattice geometries, each tailored to specific needs. With these diverse lattice geometries, SpinGlassNetworks empowers users to model and study complex spin systems with a high degree of flexibility and precision. 

## Super square lattice
The `super_square_lattice` geometry represents a square lattice where unit cells consist of multiple spins, offering the unique feature of accommodating diagonal next-nearest-neighbor interactions. This geometry allows for a nuanced exploration of spin interactions beyond the traditional square lattice framework.

```@docs
super_square_lattice
```

```@example
using SpinGlassEngine, SpinGlassNetworks, LabelledGraphs

# load Chimera instance and create Ising graph
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"
ig = ising_graph(instance)

# Loaded instance is zephyr graph
m = 5
n = 5
t = 4

cl_h = clustered_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((m, n, t))
)

println("Number of nodes in oryginal instance: ", length(LabelledGraphs.vertices(ig)), "\n", " Number of nodes in clustered Hamiltonian: ", length(LabelledGraphs.vertices(cl_h)))
```

## Pegasus graphs
The Pegasus graph is a type of graph architecture used in quantum computing systems, particularly in the quantum annealing machines developed by D-Wave Systems. It is designed to provide a grid of qubits with specific connectivity patterns optimized for solving certain optimization problems. Futer details can be found [here](https://docs.dwavesys.com/docs/latest/c_gs_4.html#pegasus-graph).
```@raw html
<img src="../images/peg.pdf" width="200%" class="center"/>
```

```@docs
pegasus_lattice
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

println("Number of nodes in original instance: ", length(LabelledGraphs.vertices(ig)), "\n", " Number of nodes in clustered Hamiltonian: ", length(LabelledGraphs.vertices(cl_h)))
```


## Zephyr graphs
The Zephyr graph is a type of graph architecture used in quantum computing systems, particularly in the quantum annealing machines developed by D-Wave Systems. Futer details can be found [here](https://docs.dwavesys.com/docs/latest/c_gs_4.html#zephyr-graph).
```@raw html
<img src="../images/zep.pdf" width="200%" class="center"/>
```

```@docs
zephyr_lattice
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

println("Number of nodes in oryginal instance: ", length(LabelledGraphs.vertices(ig)), "\n", " Number of nodes in clustered Hamiltonian: ", length(LabelledGraphs.vertices(cl_h)))
```