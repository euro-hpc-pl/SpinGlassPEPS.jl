EXACT_ENERGIES = [
    -1.75, -1.75, -1.55, -1.55, -1.45, -1.45, -1.25, -1.25, -0.75, -0.75, -0.55,
    -0.55, -0.45, -0.45, -0.25, -0.25,  0.05,  0.05,  0.25,  0.25,  0.75,  0.75,
     0.95,  0.95, 1.05,  1.05, 1.25,  1.25, 1.75,  1.75,  1.95,  1.95,
]

@testset "Pegasus-like (smallest cross-square-star) instance has the correct solution" begin
    m, n, t = 2, 3, 1
    L = n * m * t

    β = 1.0
    bond_dim = 16
    num_states = 2 ^ L

    instance = "$(@__DIR__)/instances/pathological/cross_3_2.txt"

    ig = ising_graph(instance)

    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = 0.0)

    for Strategy ∈ (SVDTruncate, Zipper), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
            for transform ∈ (all_lattice_transformations[4], )

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

                sol, s = low_energy_spectrum(ctr, search_params)
                @test EXACT_ENERGIES ≈ sol.energies

                ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                @test sol.energies ≈ energy.(Ref(ig), ig_states)

                potts_h_states = decode_state.(Ref(net), sol.states)
                @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))

                clear_memoize_cache()
            end
        end
    end
end
