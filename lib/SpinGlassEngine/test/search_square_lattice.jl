using SpinGlassExhaustive

function bench(instance::String)
    m, n, t = 5, 5, 4

    eng_sbm = -215.2368
    β = 1.0
    bond_dim = 12
    δp = 1E-4
    num_states = 20

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = δp)
    Gauge = NoUpdate
    graduate_truncation = true
    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, Zipper), transform ∈ all_lattice_transformations

        for Layout ∈ (GaugesEnergy, EnergyGauges, EngGaugesEng), Sparsity ∈ (Dense, Sparse)
            net = PEPSNetwork{KingSingleNode{Layout},Sparsity}(m, n, potts_h, transform)
            ctr = MpsContractor{Strategy,Gauge}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = graduate_truncation,
            )
            sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            push!(energies, sol_peps.energies)
            @test energies[begin][1] ≈ eng_sbm
            clear_memoize_cache()
        end
    end
end

# bench("$(@__DIR__)/instances/square_diagonal/square_5x5.txt")
bench("$(@__DIR__)/instances/square_diagonal/diagonal_5x5.txt")
