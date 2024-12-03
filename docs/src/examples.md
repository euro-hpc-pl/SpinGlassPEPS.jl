# More examples
In this section, we present a few examples demonstrating the functionality of `SpinGlassPEPS.jl`. These examples showcase how to construct tensor networks, work with Potts Hamiltonians, and perform branch and bound search for optimization problems.

The full list of examples, including detailed code and visualizations, can be found on the [GitHub](https://github.com/euro-hpc-pl/SpinGlassPEPS.jl/tree/master/examples).

## Inpainting
This example demonstrates how to solve an [inpainting](https://en.wikipedia.org/wiki/Inpainting) problem  using `SpinGlassPEPS.jl`. 

```@julia
using SpinGlassPEPS

function bench_inpaining(::Type{T}, β::Real, max_states::Integer, bond_dim::Integer) where {T}
	potts_h= potts_hamiltonian(instance, 120, 120)

	params = MpsParameters{T}(; bond_dim = bond_dim, method = :svd)
	search_params = SearchParameters(; max_states = max_states)
	net = PEPSNetwork{SquareSingleNode{GaugesEnergy}, Sparse, T}(120, 120, potts_h, rotation(0))
	ctr = MpsContractor{SVDTruncate, NoUpdate, T}(net, params; onGPU = true, beta = convert(Float64, β), graduate_truncation = true)
    droplets = SingleLayerDroplets(; max_energy = 100, min_size = 100 , metric = :hamming, mode=:RMF)
	merge_strategy = merge_branches(ctr; merge_prob = :none, droplets_encoding = droplets)

	sol, info = low_energy_spectrum(ctr, search_params, merge_strategy)
    ground = sol.energies[begin]

    println("Best energy found: $(ground)")
    sol
end
sol = bench_inpaining(Float64, 6, 64, 4)
```
### Key steps
* The function `potts_hamiltonian` generates a Hamiltonian for a 120x120 grid.
* A PEPS tensor network is initialized with the RMF instance, using `Sparse` tensors and a `SquareSingleNode` representation.
* The contraction is performed using parameters saved in `MpsContractor`. We use `SVDTruncate` to optimize boundary MPSs. The `onGPU` flag enables GPU acceleration.
* The function `low_energy_spectrum` searches for low-energy configurations with branch merging and droplet search.

!!! info "Data visualisation"
    To fully interpret the results, the data might be visualized. This example generates raw data, but a full visualization guide is available on [GitHub](https://github.com/euro-hpc-pl/SpinGlassPEPS.jl/blob/master/examples/inpaining.jl)

## D-Wave Pegasus
This example demonstrates how to perform an optimization on the Pegasus lattice with 216 spins using `SpinGlassPEPS.jl`. It is a computationally demanding example that involves truncating cluster states from 2^24 to the 2^16  most probable states. The example requires GPU support for efficient computation and may take several minutes to complete.

```@julia
using SpinGlassPEPS

function run_pegasus_bench(::Type{T}; topology::NTuple{3, Int}) where {T}
    m, n, t = topology
    instance = "$(@__DIR__)/instances/P4_CBFM-P.txt"
    results_folder = "$(@__DIR__)/lbp"
    isdir(results_folder) || mkdir(results_folder)

    lattice = pegasus_lattice(topology)

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        spectrum = full_spectrum,
        cluster_assignment_rule = lattice,
    )

    potts_h = truncate_potts_hamiltonian(potts_h, T(2), 2^16, results_folder, "P4_CBFM-P"; tol=1e-6, iter=2)

    params = MpsParameters{T}(bond_dim=16, num_sweeps=1)
    search_params = SearchParameters(max_states=2^8, cutoff_prob=1e-4)

    best_energies = T[]

    for transform in all_lattice_transformations
        net = PEPSNetwork{SquareCrossDoubleNode{GaugesEnergy}, Sparse, T}(m, n, potts_h, transform)
        ctr = MpsContractor(Zipper, net, params; onGPU=true, beta=T(2), graduate_truncation=true)

        droplets = SingleLayerDroplets(max_energy=10, min_size=54, metric=:hamming)
        merge_strategy = merge_branches(ctr; merge_prob=:none, droplets_encoding=droplets)

        sol, _ = low_energy_spectrum(ctr, search_params, merge_strategy)
        sol2 = unpack_droplets(sol, T(2))

        println("Droplet energies: $(sol2.energies)")

        push!(best_energies, sol.energies[1])
        clear_memoize_cache()
    end

    ground = best_energies[1]
    @assert all(ground .≈ best_energies)
    println("Best energy found: $(best_energies[1])")
end

T = Float64
@time run_pegasus_bench(T; topology = (3, 3, 3))
```

### Key steps
* The Potts Hamiltonian is constructed from an Ising graph build based on the input file (`P4_CBFM-P.txt`).
* The Hamiltonian is then truncated using a truncate_potts_hamiltonian function, keeping only the most probable states, with the number of states in every cluster reduced from 2^24 to 2^16 . This truncation significantly reduces the numerical cost of the subsequent tensor network contraction. Note that, this is optional step.
* A PEPS (Projected Entangled Pair States) network is created with a `SquareCrossDoubleNode` and `GaugesEnergy` for representing the system on the lattice. 
* Structures such as `MpsParameters` and  `SearchParameters` stores the information about e.g. a predefined bond dimension and a number of states considered during the search.
* `MpsContractor` stores information needed in subsequent tensor network contraction. The contraction is performed using a `Zipper` method for optimizing boundary MPS. 
* The function `low_energy_spectrum` searches for low-energy configurations with branch merging and droplet search. We are searching for droplets with an energy `max_energy` higher than the ground state by at most 10, and the minimal size `min_size` of 54.