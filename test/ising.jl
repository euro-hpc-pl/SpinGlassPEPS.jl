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

        states, energies = brute_force(ig, k)

        s = 5
        display(states[1:s])
        println("   ")
        display(energies[1:s])
        println("   ")

        @test energies ≈ energy.(states, Ref(ig))

        _states, _energies = SpinGlassPEPS._brute_force(ig, k)

        @test _energies ≈ energies
        @test _states == states

        set_prop!(ig, :β, rand(Float64))
        
        ρ = gibbs_tensor(ig)

        @test size(ρ) == Tuple(fill(2, N))

        β = get_prop(ig, :β)

        r = exp.(-β .* energies)
        R = r ./ sum(r)

        @test sum(R) ≈ 1
        @test sum(ρ) ≈ 1        

        @test [ ρ[idx.(σ)...] for σ ∈ states ] ≈ R
    end

    @testset "Naive brute force for general spins" begin
        L = 4 
        instance = "$(@__DIR__)/instances/$(L)_001.txt"  

        ig = ising_graph(instance, L)

        set_prop!(ig, :rank, (3,2,5,4))
        rank = get_prop(ig, :rank)

        all = prod(rank)
        states, energies = brute_force(ig, all)

        β = get_prop(ig, :β)
        ρ = exp.(-β .* energies)

        ϱ = ρ ./ sum(ρ) 
        ϱ̃ = gibbs_tensor(ig)

        @test [ ϱ̃[idx.(σ)...] for σ ∈ states ] ≈ ϱ 
    end

    @testset "Reading from Dict" begin
        instance_dict = Dict()
        ising = CSV.File(instance, types=[Int, Int, Float64], comment = "#")

        for (i, j, v) ∈ ising
            push!(instance_dict, (i, j) => v)
        end

        ig = ising_graph(instance, N)
        ig_dict = ising_graph(instance_dict, N)

        @test gibbs_tensor(ig) ≈ gibbs_tensor(ig_dict)
    end 
end