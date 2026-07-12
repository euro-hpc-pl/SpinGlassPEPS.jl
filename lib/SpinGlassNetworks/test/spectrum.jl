using CSV
using LinearAlgebra
using LabelledGraphs

@testset "Spectrum and related properties of the Ising model are correct" begin
    L = 4
    N = L^2
    instance = "$(@__DIR__)/instances/$(N)_001.txt"
    ig = ising_graph(instance)

    @testset "Naive brute force for +/-1" begin
        k = 2^N
        sp = brute_force(ig, num_states = k)

        β = rand(Float64)
        ρ = gibbs_tensor(ig, β)

        r = exp.(-β .* sp.energies)
        R = r ./ sum(r)

        @test size(ρ) == Tuple(fill(2, N))
        @test sum(R) ≈ sum(ρ) ≈ 1
        @test sp.energies ≈ energy(sp.states, ig)
        @test [ρ[idx.(σ)...] for σ ∈ sp.states] ≈ R

        for (i, state) in enumerate(sp.states)
            state_dict = Dict(i => s for (i, s) ∈ enumerate(state))
            energy(ig, state_dict) ≈ sp.energies[i]
        end
    end

    @testset "Naive brute force for general spins" begin
        L = 4
        ig = ising_graph("$(@__DIR__)/instances/$(L)_001.txt")

        set_prop!(ig, :rank, [3, 2, 5, 4])
        rank = get_prop(ig, :rank)

        all = prod(rank)
        sp = full_spectrum(ig, num_states = all)

        β = rand(Float64)
        ρ = exp.(-β .* sp.energies)

        ϱ = ρ ./ sum(ρ)
        ϱ̃ = gibbs_tensor(ig, β)

        @test [ϱ̃[idx.(σ)...] for σ ∈ sp.states] ≈ ϱ
    end
end
