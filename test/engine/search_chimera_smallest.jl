@testset "Smallest chimera pathological instance has the correct spectrum for all transformations" begin
    exact_energies = [-2.6, -1.1, -0.6, -0.4, -0.4, 1.1, 1.9, 2.1]

    m, n, t = 3, 1, 1
    L = n * m * t

    β = 1.0
    bond_dim = 16
    num_states = 2^8
    instance = "$(@__DIR__)/instances/pathological/chim_$(n)_$(m)_$(t).txt"

    ig = ising_graph(instance)
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    search_params = SearchParameters(; max_states = num_states, cutoff_prob = 0.0)
    Gauge = NoUpdate
    for T in [Float32, Float64]
        energies = Vector{T}[]
        params = MpsParameters{T}(; bond_dim = bond_dim, var_tol = T(1E-8), num_sweeps = 4)
        for Strategy ∈ (SVDTruncate, Zipper),
            Sparsity ∈ (Dense, Sparse),
            Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng),
            transform ∈ all_lattice_transformations

            net = PEPSNetwork{SquareSingleNode{Layout},Sparsity,T}(m, n, potts_h, transform)
            ctr = MpsContractor{Strategy,Gauge,T}(
                net,
                params;
                onGPU = onGPU,
                beta = T(β),
                graduate_truncation = true,
            )
            sol, s = low_energy_spectrum(ctr, search_params)
            @test eltype(sol.energies) == T
            @test sol.energies ≈ exact_energies

            ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
            @test sol.energies ≈ energy.(Ref(ig), ig_states)
            potts_h_states = decode_state.(Ref(net), sol.states)
            @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

            norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
            @test isapprox(
                norm_prob,
                exp.(-β .* (sol.energies .- sol.energies[1])),
                atol = eps(T),
            )

            push!(energies, sol.energies)
            clear_memoize_cache()
        end
        @test all(e -> e ≈ first(energies), energies)
    end
end
