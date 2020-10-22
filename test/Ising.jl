using MetaGraphs
using LightGraphs
using GraphPlot

@testset "Ising" begin

    L = 3
    instance = "./lattice_$L.txt"    
    ig = ising_graph(instance, L^2)

    E = get_prop(ig, :energy)

    println(ig)
    println("energy: $E")

    for spin ∈ vertices(ig)
        println("neighbors of spin $spin are: ", neighbors(ig, spin) )
    end

    @test nv(ig) == L^2

    for i ∈ 1:nv(ig)
        @test has_vertex(ig, i)
    end    

    #gplot(ig, nodelabel=1:L^2)
end