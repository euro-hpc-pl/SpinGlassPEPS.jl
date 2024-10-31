# SpinGlassPEPS.jl 

| **Documentation** | **Digital Object Identifier** |
|:-----------------:|:-----------------------------:|
|[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://euro-hpc-pl.github.io/SpinGlassPEPS.jl/dev/)| TO DO |


<p style="text-align: justify;">Welcome to `SpinGlassPEPS.jl`, an open-source Julia package designed for heuristically finding low-energy configurations of generalized Potts models, including Ising and QUBO (Quadratic Unconstrained Binary Optimization) problems. It utilizes heuristic tensor network contraction algorithms on quasi-2D geometries, such as the graphs describing the structure of the D-Waves QPU processor.</p>

> **_IMPORTANT:_**  Right now, this package supports only julia 1.10

## Package Description

<p style="text-align: justify;">This package combines advanced heuristics to address optimization challenges and employs tensor network contractions to compute conditional probabilities to identify the most probable states according to the Gibbs distribution. `SpinGlassPEPS.jl` is a tool for reconstructing the low-energy spectrum of Ising spin glass Hamiltonians and RMF Hamiltonians. Beyond energy computations, the package offers insights into spin configurations, associated probabilities, and retains the largest discarded probability during the branch and bound optimization procedure. Notably, `SpinGlassPEPS.jl` goes beyond ground states, introducing a unique feature for identifying and analyzing spin glass droplets — collective excitations crucial for understanding system dynamics beyond the fundamental ground state configurations.</p>


## Package architecture
The package `SpinGlassPEPS.jl` includes:

* `SpinGlassTensors.jl` - <p style="text-align: justify;">Package containing  essential tools for creating and manipulating tensors that constitute the PEPS network, with support for both CPU and GPU utilization. It manages core operations on tensor networks, including contraction, using the boundary Matrix Product State approach. This package primarily functions as a backend, and users generally do not interact with it directly.</p>

* `SpinGlassNetworks.jl` - <p style="text-align: justify;">Package  facilitating the generation of an Ising graph from a given instance using a set of standard inputs (e.g., instances compatible with the Ocean environment provided by D-Wave) and suports clustering to create effective Potts Hamiltonians</p>

* `SpinGlassEngine.jl` - <p style="text-align: justify;">The main package , consisting of routines for executing the branch-and-bound method (with the ability to leverage the problem’s locality) for a given Potts instance. It also includes capabilities for reconstructing the low-energy spectrum from identified localized excitations and provides a tensor network constructor.<p>


# Code Example

Breakdown of this example can be found in the documentation.

```@julia
using SpinGlassPEPS

function get_instance(topology::NTuple{3, Int})
    m, n, t = topology
    "$(@__DIR__)/instances/square_diagonal/$(m)x$(n)x$(t).txt"
end

function run_square_diag_bench(::Type{T}; topology::NTuple{3, Int}) where {T}
    m, n, _ = topology
    instance = get_instance(topology)
    lattice = super_square_lattice(topology)

    hamming_dist = 5
    eng = 10

    best_energies = T[]

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        spectrum = full_spectrum,
        cluster_assignment_rule = lattice,
    )

    params = MpsParameters{T}(; bond_dim = 16, num_sweeps = 1)
    search_params = SearchParameters(; max_states = 2^8, cut_off_prob = 1E-4)

    for transform ∈ all_lattice_transformations
        net = PEPSNetwork{KingSingleNode{GaugesEnergy}, Dense, T}(
            m, n, potts_h, transform,
        )

        ctr = MpsContractor(SVDTruncate, net, params; 
            onGPU = false, beta = T(2), graduate_truncation = true,
        )

        single = SingleLayerDroplets(eng, hamming_dist, :hamming)
        merge_strategy = merge_branches(
            ctr; merge_type = :nofit, update_droplets = single,
        )

        sol, _ = low_energy_spectrum(ctr, search_params, merge_strategy)

        push!(best_energies, sol.energies[1])
        clear_memoize_cache()
    end

    ground = best_energies[1]
    @assert all(ground .≈ best_energies)

    println("Best energy found: $(ground)")
end

T = Float64
@time run_square_diag_bench(T; topology = (3, 3, 2))
```


# Citing

Article decribing this package and code
```
@article{SpinGlassPEPS.jl,
    author = {Tomasz \'{S}mierzchalski and Anna Maria Dziubyna and Konrad Ja\l{}owiecki and Zakaria
    Mzaouali and {\L}ukasz Pawela and Bart\l{}omiej Gardas and Marek M. Rams},
    title = {{SpinGlassPEPS.jl}: low-energy solutions for near-term quantum annealers},
    journal = {},
    year = {},
}
```

Article describing in detail used algorithms and containing extensive benchmarks
```
@article{SpinGlassPEPS, 
    author = {Anna Maria Dziubyna and Tomasz \'{S}mierzchalski and Bart\l{}omiej Gardas and Marek M. Rams and Masoud Mohseni},
    title = {Limitations of tensor network approaches for optimization and sampling: A comparison against quantum and classical {Ising} machines},
    journal = {},
    year = {},
}
```

Those citations are also in [`CITATION.bib`](CITATION.bib).