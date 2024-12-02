using SpinGlassPEPS
using Logging

disable_logging(LogLevel(1))


function get_instance(topology::NTuple{3, Int})
    m, n, t = topology
    "$(@__DIR__)/instances/$(m)x$(n)x$(t).txt"
end


function run_square_diag_bench(::Type{T}; topology::NTuple{3, Int}) where {T}
    m, n, _ = topology
    instance = get_instance(topology)
    lattice = super_square_lattice(topology)

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

        droplets = SingleLayerDroplets(; max_energy = 10, min_size = 5, metric = :hamming)
        merge_strategy = merge_branches(
            ctr; merge_prob = :none , droplets_encoding = droplets,
        )

        sol, _ = low_energy_spectrum(ctr, search_params, merge_strategy)
        droplets = unpack_droplets(sol, T(2))
        ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), droplets.states)
        ldrop = length(droplets.states)

        println("Number of droplets for transform $(transform) is $(ldrop)")

        push!(best_energies, sol.energies[1])
        clear_memoize_cache()
    end

    ground = best_energies[1]
    @assert all(ground .≈ best_energies)

    println("Best energy found: $(ground)")
end


T = Float64
@time run_square_diag_bench(T; topology = (3, 3, 2))
