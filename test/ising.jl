using MetaGraphs
using LightGraphs
using GraphPlot
using Base

@testset "Ising" begin

    L = 3
    N = L^2
    instance = "./lattice_$L.txt"    
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
end