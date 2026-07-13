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

    params = MpsParameters{Float64}(; bond_dim = bond_dim, var_tol = 1E-8, num_sweeps = 4)
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = 0.0)
    Gauge = NoUpdate

    energies = Vector{Float64}[]
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
                    merge_branches(
                        ctr;
                        merge_prob = :none,
                        droplets_encoding = SingleLayerDroplets(;
                            max_energy = 2.2,
                            min_size = 1,
                            metric = :hamming,
                        ),
                    ),
                )

                @test sol1.energies ≈ exact_energies[[1]]
                sol2 = unpack_droplets(sol1, β)
                @test sol2.energies ≈ exact_energies[1:5]

                for sol ∈ (sol1, sol2)
                    ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol.states)
                    @test sol.energies ≈ energy.(Ref(ig), ig_states)

                    potts_h_states = decode_state.(Ref(net), sol.states)
                    @test sol.energies ≈ energy.(Ref(potts_h), potts_h_states)

                    norm_prob = exp.(sol.probabilities .- sol.probabilities[1])
                    @test norm_prob ≈ exp.(-β .* (sol.energies .- sol.energies[1]))
                end
                clear_memoize_cache()
            end
        end
    end
end
