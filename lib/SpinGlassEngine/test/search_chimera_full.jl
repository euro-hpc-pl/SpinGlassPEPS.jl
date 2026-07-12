using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine


function bench(instance::String)
    m, n, t = 16, 16, 8

    max_cl_states = 2^(t - 0)

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 32
    dE = 3.0
    δp = exp(-β * dE)
    num_states = 500

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        max_cl_states,
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    params = MpsParameters{Float64}(;
        bond_dim = bond_dim,
        var_tol = 1E-8,
        num_sweeps = 4,
        tol_SVD = 1E-16,
    )
    search_params = SearchParameters(; max_states = num_states, cutoff_prob = δp)

    energies = Vector{Float64}[]
    for Strategy ∈ (SVDTruncate, Zipper), Sparsity ∈ (Dense, Sparse)
        for Gauge ∈ (NoUpdate, GaugeStrategy, GaugeStrategyWithBalancing)
            for Layout ∈ (GaugesEnergy,), transform ∈ all_lattice_transformations

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
                sol, s = low_energy_spectrum(
                    ctr,
                    search_params,
                    merge_branches(ctr; merge_prob = :none),
                )

                @test sol.energies[begin] ≈ ground_energy

                push!(energies, sol.energies)
                clear_memoize_cache()
            end
        end
    end
    @test all(e -> e ≈ first(energies), energies)
end

bench("$(@__DIR__)/instances/chimera_droplets/2048power/001.txt")
