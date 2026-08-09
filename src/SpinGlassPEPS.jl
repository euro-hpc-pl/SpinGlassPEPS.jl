module SpinGlassPEPS

using Reexport

include("tensors/SpinGlassTensors.jl")
include("networks/SpinGlassNetworks.jl")
include("exhaustive/SpinGlassExhaustive.jl")
include("engine/SpinGlassEngine.jl")

# Bring every submodule's exports into scope (for the module body, the docs
# @docs blocks, and cross-submodule use) but re-export only the curated public
# 2.0 API listed below. Power users can still reach the full internal surface
# via `using SpinGlassPEPS.SpinGlass<Tensors|Networks|Exhaustive|Engine>`.
using .SpinGlassTensors
using .SpinGlassNetworks
using .SpinGlassExhaustive
using .SpinGlassEngine

export
    # submodules — direct access to the full internal API for advanced use
    SpinGlassTensors, SpinGlassNetworks, SpinGlassExhaustive, SpinGlassEngine,
    # Ising / Potts model construction
    ising_graph, IsingGraph, potts_hamiltonian, PottsHamiltonian,
    decode_potts_hamiltonian_state,
    # lattice cluster-assignment geometries
    super_square_lattice, pegasus_lattice, zephyr_lattice,
    # spectra & exhaustive solvers
    full_spectrum, brute_force, Spectrum, exhaustive_search, exhaustive_search_bucket,
    # QUBO / random-graph helpers
    generate_random_graph, graph_to_dict, graph_to_qubo, energy_qubo, kernel, kernel_qubo,
    # tensor-network node geometries, sparsity, layout, gauge strategy
    SquareSingleNode, SquareDoubleNode, SquareCrossDoubleNode, KingSingleNode,
    Dense, Sparse, GaugesEnergy, EnergyGauges, EngGaugesEng, NoUpdate,
    # network, contractor and their parameters
    PEPSNetwork, MpsContractor, MpsParameters, SearchParameters,
    # boundary-MPS contraction strategies
    SVDTruncate, Zipper,
    # lattice transformations
    LatticeTransformation, rotation, reflection, all_lattice_transformations,
    # branch-and-bound low-energy search
    low_energy_spectrum, Solution, merge_branches, gibbs_sampling,
    # concurrent sweep over lattice transformations + device-memory governor
    sweep_transformations, SweepSolution, SweepReport, TransformReport,
    best_solution, DeviceBudget,
    # inverse-temperature ladder with warm-started boundary MPS
    beta_ladder, set_beta!, BetaLadderSolution, BetaStepReport, selected_solution,
    # contraction error control
    TruncationLog, TruncationStats, truncation_stats, TRUNCATION_LOG,
    DEVICE_MEMORY_BUDGET,
    # droplets (low-energy excitations)
    SingleLayerDroplets, NoDroplets, Droplet, Droplets, Flip, unpack_droplets,
    # belief-propagation dimensional reduction
    belief_propagation, potts_hamiltonian_2site, truncate_potts_hamiltonian,
    # Hamiltonian inspection accessors
    cluster_spectrum, biases, couplings,
    # documented core types
    MpoTensor, QMps, QMpo, PoolOfProjectors,
    # deprecated no-op compatibility shims (kept one release cycle)
    clear_memoize_cache, clear_memoize_cache_after_row

using PrecompileTools: @setup_workload, @compile_workload

# Precompile the end-to-end CPU solve path so the first real solve in a fresh
# session does not pay full compilation for the whole type stack (the cold-start
# cost measured after the type-parameter consolidation). Uses a 3-spin instance
# (the smallest tested case) on the CPU path — no GPU, no external files.
@setup_workload begin
    inst = Dict((1, 1) => 1.0, (2, 2) => 0.75, (3, 3) => -0.5, (1, 2) => 0.25, (2, 3) => 0.6)
    @compile_workload begin
        ig = ising_graph(inst)
        ph = potts_hamiltonian(
            ig;
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((3, 1, 1)),
        )
        net = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
            3, 1, ph, rotation(0),
        )
        ctr = MpsContractor{SVDTruncate,NoUpdate,Float64}(
            net,
            MpsParameters{Float64}(; bond_dim = 8, num_sweeps = 1);
            onGPU = false, beta = 1.0, graduate_truncation = true,
        )
        low_energy_spectrum(
            ctr,
            SearchParameters(; max_states = 8, cutoff_prob = 0.0);
            show_progress = false,
        )
    end
end

end
