# Constructing PEPS tensor network

After creating the Potts Hamiltonian, we can turn it into a PEPS tensor network as shown in the Section [Brief description of the algorithm](../algorithm.md). 

```@docs
PEPSNetwork
```

## Basic example of usage

```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks
using SpinGlassExhaustive

m, n, t = 5, 5, 4
onGPU = true
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"

Strategy = Zipper
transform = rotation(0)
Layout = GaugesEnergy
Sparsity = Sparse
R = Float64

ig = ising_graph(instance)
potts_h = potts_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

net = PEPSNetwork{KingSingleNode{Layout}, Sparsity, R}(m, n, potts_h, transform)
```
