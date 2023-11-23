# Constructing PEPS tensor network

After creating the clustered Hamiltonian, we can turn it into a PEPS tensor network as shown in the figure below. 

```@docs
PEPSNetwork
```

## Basic example of usage

```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

m, n, t = 5, 5, 4
onGPU = true
instance = "$(@__DIR__)/../../src/instances/square_diagonal/5x5/diagonal.txt"

Strategy = Zipper
transform = rotation(0)
Layout = GaugesEnergy
Sparsity = Sparse

ig = ising_graph(instance)
cl_h = clustered_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

net = PEPSNetwork{SquareCrossSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
```
