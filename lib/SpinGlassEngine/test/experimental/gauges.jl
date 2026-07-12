using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t - 0)

β = 0.5
bond_dim = 32
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

potts_h = potts_hamiltonian(
    ising_graph(instance),
    max_cl_states,
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)

params = MpsParameters{Float64}(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

# @testset "Overlaps calculated differently are the same." begin
# for Lattice ∈ (SquareSingleNode, KingSingleNode) 
#     for Sparsity ∈ (Dense, Sparse), transform ∈ all_lattice_transformations[[1]]
#         for Layout ∈ (GaugesEnergy, EnergyGauges, EngGaugesEng)
#             net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, potts_h, transform, :id)
#             ctr_svd = MpsContractor{SVDTruncate, GaugeStrategy}(net, [β/8, β/4, β/2, β], true, params; onGPU=onGPU)
#             ctr_anneal = MpsContractor{MPSAnnealing, GaugeStrategy}(net, [β/8, β/4, β/2, β], true, params; onGPU=onGPU)

#             @testset "Overlaps calculated for different Starategies are the same." begin
#                 for i ∈ 1:m-1
#                     ψ_top = mps_top(ctr_svd, i)
#                     ϕ_top = mps_top(ctr_anneal, i)
#                     @test ψ_top * ψ_top ≈ 1
#                     @test ϕ_top * ϕ_top ≈ 1
#                     @test ψ_top * ϕ_top ≈ 1
#                 end
#             end
#         end
#         clear_memoize_cache()

#         for Layout ∈ (GaugesEnergy,)
#             net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, potts_h, transform, :id)
#             ctr_svd = MpsContractor{SVDTruncate, GaugeStrategy}(net, [β/8, β/4, β/2, β], true, params; onGPU=onGPU)
#             @testset "Overlaps calculated in Python are the same as in Julia." begin
#                 overlap_python = [0.2637787707674837, 0.2501621729619047, 0.2951954406837012]

#                 for i ∈ vcat(1:m-1)#, m-1:-1:1)
#                     ψ_top = mps_top(ctr_svd, i)
#                     ψ_bot = mps(ctr_svd, i+1)
#                     overlap1 = ψ_top * ψ_bot
#                     @test isapprox(overlap1, overlap_python[i], atol=1e-5)
#                 end
#                 clear_memoize_cache()
#             end
#         end
#     end
# end
# end

@testset "Updating gauges works correctly." begin
    for Strategy ∈ (SVDTruncate,), Sparsity ∈ (Dense, Sparse)
        for Layout ∈ (GaugesEnergy,)
            for Gauge ∈ (GaugeStrategy,)
                for Lattice ∈ (SquareSingleNode, KingSingleNode),
                    transform ∈ all_lattice_transformations

                    net = PEPSNetwork{Lattice{Layout},Sparsity,Float64}(
                        m,
                        n,
                        potts_h,
                        transform,
                        :id,
                    )
                    ctr = MpsContractor{Strategy,Gauge,Float64}(
                        net,
                        params;
                        onGPU = onGPU,
                        beta = β,
                        graduate_truncation = true,
                    )

                    @testset "Overlaps calculated differently are the same." begin
                        for i ∈ 1:m-1

                            ψ_top = mps_top(ctr, i)
                            ψ_bot = mps(ctr, i + 1)

                            try
                                overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, i))
                                @test overlap ≈ ψ_bot * ψ_top
                            catch
                                println(Strategy, " ", Sparsity, " ", Lattice, " ", i)
                                overlap = Inf
                                @test overlap ≈ ψ_bot * ψ_top

                            end

                        end
                    end
                    clear_memoize_cache()

                    # @testset "ψ_bot and ψ_top are not updated in place though memoize!" begin
                    #     for aba in 1:3, i ∈ 1:m-1
                    #         println(aba," ", i)
                    #         ψ_top = mps_top(ctr, i)
                    #         ψ_bot = mps(ctr, i+1)

                    #         overlap_old = ψ_top * ψ_bot

                    #         update_gauges!(ctr, i, Val(:down))

                    #         # assert that ψ_bot and ψ_top are not updated in place though memoize!
                    #         overlap_old2 = ψ_bot * ψ_top

                    #         @test overlap_old ≈ overlap_old2


                    #     end
                    # end
                    # break
                    # clear_memoize_cache()

                    # @testset "Updating gauges from top and bottom gives the same energy." begin
                    #     update_gauges!(ctr, m, Val(:down))
                    #     sol_l, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
                    #     clear_memoize_cache()
                    #     update_gauges!(ctr, m, Val(:up))
                    #     sol_r, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
                    #     @test sol_l.energies[begin] ≈ sol_r.energies[begin]
                    # end
                    # clear_memoize_cache()
                end
            end
        end
    end
end
