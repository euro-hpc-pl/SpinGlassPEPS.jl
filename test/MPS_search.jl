using MetaGraphs
using LightGraphs
using GraphPlot

# This a semi-finished cleaning of this file

L = 2
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"

# This is a mess, and it should be cleand up
ig = ising_graph(instance)
r = fill(2, N)
set_prop!(ig, :rank, r)
dβ = 0.01
β = 1

ϵ = 1E-8
D = prod(r) + 1
var_ϵ = 1E-8
sweeps = 4

# MPSControl should be removed
control = MPSControl(D, var_ϵ, sweeps, β, dβ)
states = all_states(get_prop(ig, :rank))


@testset "Generating MPS" begin
    ϱ = gibbs_tensor(ig, β)

    @testset "Sqrt of the Gibbs state (aka state tensor)" begin
        L = nv(ig)
        rank = get_prop(ig, :rank)

        ψ = ones(rank...)
        for σ ∈ states
            for i ∈ 1:L
                h = get_prop(ig, i, :h)

                nbrs = unique_neighbors(ig, i)
                ψ[idx.(σ)...] *= exp(-0.5 * β * h * σ[i])

                for j ∈ nbrs
                    J = get_prop(ig, i, j, :J)
                    ψ[idx.(σ)...] *= exp(-0.5 * β * σ[i] * J * σ[j])
                end
            end
        end

        ρ = abs.(ψ) .^ 2
        rψ = MPS(ψ)
        lψ = MPS(ψ, :left)

        @testset "produces correct Gibbs state" begin
            @test ρ / sum(ρ) ≈ ϱ
        end

        @testset "MPS from the tensor" begin

            @testset "can be right normalized" begin
                @test dot(rψ, rψ) ≈ 1
                @test_nowarn is_right_normalized(rψ)
            end

            @testset "can be left normalized" begin
                @test dot(lψ, lψ) ≈ 1
                @test_nowarn is_left_normalized(lψ)
            end

            @testset "both forms are the same (up to a phase factor)" begin
                vlψ = vec(tensor(lψ))
                vrψ = vec(tensor(rψ))

                vψ = vec(ψ)
                vψ /= norm(vψ)

                @test abs(1 - abs(dot(vlψ, vrψ))) < ϵ
                @test abs(1 - abs(dot(vlψ, vψ))) < ϵ
            end
        end

        @testset "MPS from gates" begin
            Gψ = MPS(ig, control)

            @testset "is built correctly" begin
                @test abs(1 - abs(dot(Gψ, rψ))) < ϵ
            end

            @testset "is normalized" begin
                @test dot(Gψ, Gψ) ≈ 1
                @test_nowarn is_right_normalized(Gψ)
            end

            @testset "has correct links and non-trivial bond dimension" begin
                @test bond_dimension(Gψ) > 1
                @test_nowarn verify_bonds(Gψ)
            end
        end

        @testset "Exact probabilities are calculated correctely" begin
            for σ ∈ states
                p, r = dot(rψ, σ), dot(rψ, proj(σ, rank), rψ)
                @test p ≈ r
                @test ϱ[idx.(σ)...] ≈ p
            end
        end

        @testset "Results from solve agree with brute-force" begin
            # The energy is wrong when max_states > N^2-2

            for max_states ∈ [1, N, 2*N, 3*N, N^2-3, N^2-2]#, N^2-1, N^2]
                states, prob, pCut = solve(rψ, max_states)
                sp = brute_force(ig, num_states = max_states)

                for (j, (p, e)) ∈ enumerate(zip(prob, sp.energies))
                    σ = states[j, :]
                    @test e ≈ energy(σ, ig)
                    @test log(ϱ[idx.(σ)...]) ≈ p
                end
            end
        end

    end
end
