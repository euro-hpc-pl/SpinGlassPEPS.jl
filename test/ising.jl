using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Ising" begin

    L = 4
    N = L^2
    instance = "$(@__DIR__)/instances/$(N)_001.txt"

    ig = ising_graph(instance, N)

    E = get_prop(ig, :energy)

    println(ig)
    println("energy: $E")

    for spin ∈ vertices(ig)
        println("neighbors of spin $spin are: ", neighbors(ig, spin) )
    end

    @test nv(ig) == N

    for i ∈ 1:N
        @test has_vertex(ig, i)
    end

    A = adjacency_matrix(ig)
    display(Matrix{Int}(A))
    println("   ")

    B = zeros(Int, N, N)
    for i ∈ 1:N
        nbrs = unique_neighbors(ig, i)
        for j ∈ nbrs
            B[i, j] = 1
        end
    end

    @test B + B' == A

    gplot(ig, nodelabel=1:N)

    @testset "Naive brute force for +/-1" begin
        k = 2^N

        sp = brute_force(ig, num_states=k)

        s = 5
        display(sp.states[1:s])
        println("   ")
        display(sp.energies[1:s])
        println("   ")

        @test sp.energies ≈ energy.(sp.states, Ref(ig))

        # states, energies = brute_force(ig, num_states=k)

        # @test energies ≈ sp.energies
        # @test states == sp.states

        β = rand(Float64)
        ρ = gibbs_tensor(ig, β)

        @test size(ρ) == Tuple(fill(2, N))

        r = exp.(-β .* sp.energies)
        R = r ./ sum(r)

        @test sum(R) ≈ 1
        @test sum(ρ) ≈ 1

        @test [ ρ[idx.(σ)...] for σ ∈ sp.states ] ≈ R
    end

    @testset "Naive brute force for general spins" begin
        L = 4
        instance = "$(@__DIR__)/instances/$(L)_001.txt"

        ig = ising_graph(instance, L)

        set_prop!(ig, :rank, [3,2,5,4])
        rank = get_prop(ig, :rank)

        all = prod(rank)
        sp = brute_force(ig, num_states=all)

        β = rand(Float64)
        ρ = exp.(-β .* sp.energies)

        ϱ = ρ ./ sum(ρ)
        ϱ̃ = gibbs_tensor(ig, β)

        @test [ ϱ̃[idx.(σ)...] for σ ∈ sp.states ] ≈ ϱ
    end

    @testset "Reading from Dict" begin
        instance_dict = Dict()
        ising = CSV.File(instance, types=[Int, Int, Float64], header=0, comment = "#")

        for (i, j, v) ∈ ising
            push!(instance_dict, (i, j) => v)
        end

        ig = ising_graph(instance, N)
        ig_dict = ising_graph(instance_dict, N)

        @test gibbs_tensor(ig) ≈ gibbs_tensor(ig_dict)
    end
end
