@testset "Chimera-like (pathological) instance has the correct energy spectrum for all heuristics" begin
    m, n, t = 3, 4, 3

    β = 1.0
    bond_dim = 16
    num_states = 2^8
    hamming_dist = 1

    # energies
    exact_energies = [
        -16.4,
        -16.4,
        -16.4,
        -16.4,
        -16.1,
        -16.1,
        -16.1,
        -16.1,
        -15.9,
        -15.9,
        -15.9,
        -15.9,
        -15.9,
        -15.9,
        -15.6,
        -15.6,
        -15.6,
        -15.6,
        -15.6,
        -15.6,
        -15.4,
        -15.4,
    ]

    # degenerate potts_h solutions
    exact_states = [   # E =-16.4
        [
            [1, 4, 5, 1, 2, 2, 1, 1, 1, 4, 2, 1],
            [1, 4, 7, 1, 2, 2, 1, 1, 1, 4, 2, 1],
            [1, 4, 5, 1, 2, 2, 1, 1, 1, 4, 6, 1],
            [1, 4, 7, 1, 2, 2, 1, 1, 1, 4, 6, 1],
        ],
        # E =-16.1
        [
            [2, 5, 4, 1, 1, 3, 1, 1, 1, 5, 7, 1],
            [2, 5, 2, 1, 1, 3, 1, 1, 1, 5, 3, 1],
            [2, 5, 4, 1, 1, 3, 1, 1, 1, 5, 3, 1],
            [2, 5, 2, 1, 1, 3, 1, 1, 1, 5, 7, 1],
        ],
        # E = -15.9
        [
            [1, 4, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1],
            [1, 4, 3, 1, 2, 2, 1, 1, 1, 4, 2, 1],
            [1, 4, 6, 1, 2, 2, 1, 1, 1, 4, 2, 1],
            [1, 4, 3, 1, 2, 2, 1, 1, 1, 4, 6, 1],
            [1, 4, 1, 1, 2, 2, 1, 1, 1, 4, 6, 1],
            [1, 4, 6, 1, 2, 2, 1, 1, 1, 4, 6, 1],
        ],
        # E = -15.6
        [
            [2, 5, 3, 1, 1, 3, 1, 1, 1, 5, 3, 1],
            [2, 5, 3, 1, 1, 3, 1, 1, 1, 5, 7, 1],
            [2, 5, 8, 1, 1, 3, 1, 1, 1, 5, 3, 1],
            [2, 5, 6, 1, 1, 3, 1, 1, 1, 5, 7, 1],
            [2, 5, 6, 1, 1, 3, 1, 1, 1, 5, 3, 1],
            [2, 5, 8, 1, 1, 3, 1, 1, 1, 5, 7, 1],
        ],
        # E = -15.4
        [[1, 4, 7, 1, 2, 2, 1, 1, 1, 2, 6, 1], [1, 4, 5, 1, 2, 2, 1, 1, 1, 2, 6, 1]],
    ]

    deg = Dict(
        1 => 1,
        2 => 1,
        3 => 1,
        4 => 1,
        #
        5 => 2,
        6 => 2,
        7 => 2,
        8 => 2,
        #
        9 => 3,
        10 => 3,
        11 => 3,
        12 => 3,
        13 => 3,
        14 => 3,
        #
        15 => 4,
        16 => 4,
        17 => 4,
        18 => 4,
        19 => 4,
        20 => 4,
        #
        21 => 5,
        22 => 5,
    )

    ig = ising_graph("$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt")
    potts_h = potts_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
    # for Strategy ∈ (SVDTruncate, Zipper), Sparsity ∈ (Dense, Sparse)
    #     for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng)
    #         for Lattice ∈ (SquareSingleNode, KingSingleNode), transform ∈ all_lattice_transformations
    for Strategy ∈ (Zipper,), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges,)
            for transform ∈ all_lattice_transformations

                net = PEPSNetwork{SquareSingleNode{Layout},Sparsity,Float64}(
                    m,
                    n,
                    potts_h,
                    transform,
                )
                ctr = MpsContractor{Strategy,Gauge,Float64}(
                    net,
                    params;
                    onGPU = onGPU,
                    beta = β,
                    graduate_truncation = true,
                )
                sol1, s = low_energy_spectrum(
                    ctr,
                    search_params,
                    merge_branches_blur(
                        ctr,
                        hamming_dist,
                        :none,
                        SingleLayerDroplets(;
                            max_energy = 1.01,
                            min_size = 1,
                            metric = :hamming,
                        ),
                    ),
                )
                # sol1, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :none , SingleLayerDroplets(1.01, 1, :hamming)))

                @test sol1.energies ≈ [exact_energies[1]]
                sol2 = unpack_droplets(sol1, β)

                @test sol2.energies[1:length(exact_energies)] ≈ exact_energies

                for sol ∈ (sol1, sol2)
                    ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                    @test sol.energies ≈ energy.(Ref(ig), ig_states)

                    potts_h_states = decode_state.(Ref(net), sol.states)
                    @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                    norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                    @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
                end

                # for (i, σ) ∈ enumerate(sol.states) @test σ ∈ exact_states[deg[i]] end

                clear_memoize_cache()
            end
        end
    end
end
