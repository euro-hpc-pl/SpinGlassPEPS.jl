# SpinGlassPEPS.jl

| **Documentation** | **Digital Object Identifier** |
|:-----------------:|:-----------------------------:|
|[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://euro-hpc-pl.github.io/SpinGlassPEPS.jl/dev/)| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3245496.svg)](https://doi.org/10.5281/zenodo.14627393)|


<div align="justify">
Welcome to `SpinGlassPEPS.jl`, an open-source Julia package designed for heuristically finding low-energy configurations of generalized Potts models, including Ising and QUBO (Quadratic Unconstrained Binary Optimization) problems. It utilizes heuristic tensor network contraction algorithms on quasi-2D geometries, such as the graphs describing the structure of the D-Waves QPU processor.
</div>


## Installation

SpinGlassPEPS is distributed as one Julia package. From the Julia package prompt:

```julia
pkg> add SpinGlassPEPS
```

Then load the complete solver stack with:

```julia
using SpinGlassPEPS
```

## Package description

<div align="justify">
This package combines advanced heuristics to address optimization challenges and employs tensor network contractions to compute conditional probabilities to identify the most probable states according to the Gibbs distribution. `SpinGlassPEPS.jl` is a tool for reconstructing the low-energy spectrum of Ising spin glass Hamiltonians and RMF Hamiltonians. Beyond energy computations, the package offers insights into spin configurations, associated probabilities, and retains the largest discarded probability during the branch and bound optimization procedure. Notably, `SpinGlassPEPS.jl` goes beyond ground states, introducing a unique feature for identifying and analyzing spin glass droplets — collective excitations crucial for understanding system dynamics beyond the fundamental ground state configurations.
</div>

## Package architecture

The complete implementation now lives in this repository and is installed as a
single package. Its four implementation layers remain as internal modules so
their responsibilities and namespaces stay clear:

<div align="justify">

* `SpinGlassTensors` provides the tensor and boundary-MPS machinery, including CPU and GPU contractions.

* `SpinGlassNetworks` constructs Ising graphs, clustered Potts Hamiltonians, and supported lattice mappings.

* `SpinGlassExhaustive` contains CPU/GPU exhaustive solvers used for small problems and validation.

* `SpinGlassEngine` builds PEPS networks and runs branch-and-bound searches, sampling, and droplet reconstruction.
</div>

Their public APIs are re-exported by `SpinGlassPEPS`, so normal usage needs only
`using SpinGlassPEPS`. Advanced users can still qualify implementation details,
for example as `SpinGlassPEPS.SpinGlassNetworks`.

## Running the standard protocol concurrently

The solver is normally run once per lattice transformation, keeping the best
result. `sweep_transformations` replaces that hand-written loop with one call,
adds deterministic per-transformation seeding, reports how much weight the
contraction discarded and whether the independent contraction orders agreed, and —
when you ask for concurrency — rations device memory against a measured
per-solve reservation instead of assuming a fixed fan-out.

Concurrency pays on CPU — up to **1.76×** over the serial loop, and it is on by
default there — but **not on a single GPU**, where fanning the solves out measures
slower than sequencing them (0.88–0.92×) because the CUDA API and allocator
serialize this solver's many small kernels. It is therefore off by default on a
GPU. See the documentation for the measurements, including the observation that the
instances measured run faster on CPU than on the GPU at all.

```julia
sweep = sweep_transformations(
    transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{KingSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params; onGPU = true, beta = 2.0, graduate_truncation = true,
    ),
    SearchParameters(; max_states = 2^8, cutoff_prob = 1e-4),
)

sol = best_solution(sweep)
sweep.report.consensus      # how many transformations reached the best energy
sweep.report.energy_spread  # ... and how far apart the rest were
```

The documentation page "Concurrent sweeps and error control" also covers
`beta_ladder`, which walks an inverse-temperature schedule with warm-started
boundary MPS. It selects on **energy**; the truncation-error budget only excludes
rungs whose contraction is untrustworthy, and is not a ranking of solution quality.

Three runnable examples:

| file | scale | shows |
| --- | --- | --- |
| `examples/beta_ladder.jl` | 18 spins, seconds | error control and the β ladder, annotated |
| `examples/concurrent_sweep.jl` | 128 spins | the transformation sweep and its VRAM governor |
| `examples/square_50x50.jl` | 2500 spins | all three, at the scale used by the article figures |

# Code Example

A breakdown of this example can be found in the documentation. To run provided examples, activate and instantiate `Project.toml` file in "examples" folder.

```julia
using SpinGlassPEPS

function get_instance(topology::NTuple{3, Int})
    m, n, t = topology
    joinpath(pkgdir(SpinGlassPEPS), "examples", "instances", "$(m)x$(n)x$(t).txt")
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
    search_params = SearchParameters(; max_states = 2^8, cutoff_prob = 1E-4)

    for transform ∈ all_lattice_transformations
        net = PEPSNetwork{KingSingleNode{GaugesEnergy}, Dense, T}(
            m, n, potts_h, transform,
        )

        ctr = MpsContractor(SVDTruncate, net, params; 
            onGPU = false, beta = T(2), graduate_truncation = true,
        )

        single = SingleLayerDroplets(;
            max_energy = eng, min_size = hamming_dist, metric = :hamming,
        )
        merge_strategy = merge_branches(
            ctr; merge_prob = :none, droplets_encoding = single,
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

Article describing this package and code.
```
@article{SpinGlassPEPS.jl,
    author = {Tomasz \'{S}mierzchalski and Anna Maria Dziubyna and Konrad Ja\l{}owiecki and Zakaria
    Mzaouali and {\L}ukasz Pawela and Bart\l{}omiej Gardas and Marek M. Rams},
    title = {{SpinGlassPEPS.jl}: low-energy solutions for near-term quantum annealers},
    journal = {},
    year = {},
}
```

Article describing in detail used algorithms and containing extensive benchmarks.
```
@misc{SpinGlassPEPS, 
    author = {Anna Maria Dziubyna and Tomasz \'{S}mierzchalski and Bart\l{}omiej Gardas and Marek M. Rams and Masoud Mohseni},
    title = {Limitations of tensor network approaches for optimization and sampling: A comparison against quantum and classical {Ising} machines},
    year = {2024},
    eprint={2411.16431},
    archivePrefix={arXiv},
    primaryClass={cond-mat.dis-nn},
    doi = {10.48550/arXiv.2411.16431} 
}
```

Those citations are also in [`CITATION.bib`](CITATION.bib).


This project was supported by:

* The National Science Center (NCN), Poland, under Projects: Sonata Bis 10, No. 2020/38/E/ST3/00269 (T.S., Z.M.) and 2020/38/E/ST3/00150 (A.D., M.R.)
* Foundation for Polish Science (grant no POIR.04.04.00-00-14DE/18-00 carried out within the Team-Net program co-financed by the European Union under the European Regional Development Fund) (B.G., Ł.P.).
