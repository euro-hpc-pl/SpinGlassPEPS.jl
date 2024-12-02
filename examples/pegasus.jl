using SpinGlassEngine
using SpinGlassNetworks

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
