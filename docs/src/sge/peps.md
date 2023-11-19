# Constructing PEPS tensor network

After creating the clustered Hamiltonian, a visual guide to the problem's structure, we can turn it into a PEPS tensor network as shown in the figure below. In all lattices we supporrt, the essential components resembling unit cells are represented by red circles and described using the formula $\underset{x_m}{\Tr} (\exp{(-\beta E_{x_m})} P^{ml}P^{mr}P^{mu}P^{md})$. Here, $x_m$ refers to a group of spins within the cluster in focus, while $P_{ml}, P_{mr}, \ldots$ denote subsets of spins within $x_m$ interacting with neighboring clusters in various directions like left and right. Spins in adjacent clusters influence each other, depicted by blue squares. Additionally, we permit diagonal interactions between next nearest neighbors, facilitated by green squares.

```@raw html
<img src="../images/peps.pdf" width="200%" class="center"/>
```

```@docs
PEPSNetwork
```


```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

function bench(instance::String)
    m, n, t = 5, 5, 4
    onGPU = true

    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), transform ∈ all_lattice_transformations
        for Layout ∈ (GaugesEnergy, EnergyGauges, EngGaugesEng), Sparsity ∈ (Dense, Sparse)
            net = PEPSNetwork{SquareCrossSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
        end
    end
end

bench("$(@__DIR__)/../src/instances/square_diagonal/5x5/diagonal.txt")
```
