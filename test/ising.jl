using MetaGraphs
using LightGraphs
using GraphPlot
using Base

@testset "Ising" begin

    L = 4
    N = L^2 
    instance = "./instances/$(N)_001.txt"  

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

    @test B+B' == A
   
    gplot(ig, nodelabel=1:N)

    @testset "Naive brute force" begin
        k = 2^N

        states, energies = _brute_force(ig, k)

        display(states[1:5])
        println("   ")
        display(energies[1:5])
        println("   ")

        @test energies ≈ energy.(states, Ref(ig))

        states_lazy, energies_lazy = _brute_force_lazy(ig, k)

        @test energies_lazy ≈ energies
        @test states_lazy == states

        if k == 2^N

            β = rand(Float64)
            opts = GibbsControl(β, [β]) 
        
            ρ = gibbs_tensor(ig, opts)
            @test size(ρ) == Tuple(fill(2, N))

            r = exp.(-β .* energies)
            R = r ./ sum(r)

            @test sum(R) ≈ 1
            @test sum(ρ) ≈ 1        

            @test maximum(R) ≈ maximum(ρ)
            @test minimum(R) ≈ minimum(ρ)
            
            @test [ρ[_toIdx.(σ)...] for σ ∈ states] ≈ R
        end
    end
end