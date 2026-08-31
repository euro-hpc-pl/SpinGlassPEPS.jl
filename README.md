# SpinGlassPEPS.jl

| **Documentation** | **Digital Object Identifier** |
|:-----------------:|:-----------------------------:|
|[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://euro-hpc-pl.github.io/SpinGlassPEPS.jl/dev/)| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22134580.svg)](https://doi.org/10.5281/zenodo.22134580)|


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
This package uses tensor-network contractions to estimate the conditional probabilities required by a branch-and-bound search. `SpinGlassPEPS.jl` reconstructs low-energy spectra of Ising spin-glass and random Markov field Hamiltonians. It reports configurations, probabilities, and contraction diagnostics, including accumulated and maximum discarded weight. It also identifies spin-glass droplets, which are localized excitations above low-energy configurations.
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
contraction discarded and whether the distinct contraction orders agreed, and —
when you ask for concurrency — rations device memory against a measured
per-solve reservation instead of assuming a fixed fan-out.

Concurrency pays on CPU — up to **3.1×** over the serial loop, and it is on by
default there. On a GPU the `:auto` default is a conservative **1**: on the tested
RTX 5080 fanning the solves out did not beat the serial loop, while the tested H100
did benefit — the same sweeps reached **1.69×/1.44×** (c=2/c=4). Measure before
setting `concurrency = 2`–`4` explicitly on another device. See the documentation
for the measurements, including where the CPU/GPU crossover lies: the CPU leads
across the tested range except at the largest sparse case (2048 spins, bond 32),
where the GPU wins.

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
boundary MPS. A rung is trusted when `truncation.discarded_sum <= max_discarded`.
The ladder minimizes energy over trusted rungs. If none has a valid energy, it
returns the minimum-energy successful rung as an untrusted fallback. With the
default `Inf`, selection is simply by minimum energy. The discarded-weight guard
describes contraction reliability, not solution quality.

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

Article describing the package and its implementation.
```
@article{SpinGlassPEPS.jl,
    author = {Tomasz \'{S}mierzchalski and Anna Maria Dziubyna and Konrad Ja\l{}owiecki and Zakaria
    Mzaouali and {\L}ukasz Pawela and Bart\l{}omiej Gardas and Marek M. Rams},
    title = {{SpinGlassPEPS.jl}: Tensor-network package for {Ising}-like optimization on quasi-two-dimensional graphs},
    journal = {SoftwareX},
    volume = {31},
    pages = {102257},
    year = {2025},
    doi = {10.1016/j.softx.2025.102257},
}
```

Article describing the algorithms and their benchmark evaluation.
```
@article{SpinGlassPEPS, 
    author = {Anna Maria Dziubyna and Tomasz \'{S}mierzchalski and Bart\l{}omiej Gardas and Marek M. Rams and Masoud Mohseni},
    title = {Limitations of tensor-network approaches for optimization and sampling: A comparison to quantum and classical {Ising} machines},
    journal = {Physical Review Applied},
    volume = {23},
    pages = {054049},
    year = {2025},
    doi = {10.1103/PhysRevApplied.23.054049},
}
```

Those citations are also in [`CITATION.bib`](CITATION.bib).


This project was supported by:

* The National Science Center (NCN), Poland, under Projects: Sonata Bis 10, No. 2020/38/E/ST3/00269 (T.S., Z.M.) and 2020/38/E/ST3/00150 (A.D., M.R.)
* Foundation for Polish Science (grant no POIR.04.04.00-00-14DE/18-00 carried out within the Team-Net program co-financed by the European Union under the European Regional Development Fund) (B.G., Ł.P.).
