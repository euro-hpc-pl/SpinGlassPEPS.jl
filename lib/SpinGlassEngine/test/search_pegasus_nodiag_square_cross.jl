using SpinGlassExhaustive


function bench(instance::String)
    m, n, t = 4, 4, 24

    max_cl_states = 2^4

    β = 1.0
    bond_dim = 16
    δp = 1E-4
    num_states = 1000

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        max_cl_states,
        spectrum = my_brute_force,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = δp)

    # Solve using PEPS search
    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, Zipper), transform ∈ all_lattice_transformations

        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), Sparsity ∈ (Dense,)
            net = PEPSNetwork{KingSingleNode{Layout},Sparsity,Float64}(
                m,
                n,
                potts_h,
                transform,
            )
            ctr = MpsContractor{Strategy,NoUpdate,Float64}(
                net,
                params;
                onGPU = onGPU,
                beta = β,
                graduate_truncation = true,
            )
            sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            push!(energies, sol_peps.energies)
            clear_memoize_cache()
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

# best ground found for max_cl_states = 2^4: -537.0007291019799
bench("$(@__DIR__)/instances/pegasus_nondiag/4x4.txt")
