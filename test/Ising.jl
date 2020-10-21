using MetaGraphs

@testset "Ising" begin

    L = 3
    instance = "./lattice_$L.txt"    
    ig = Ising_graph(instance, L)

    E = get_prop(ig, :energy)
    
    println(ig)
    println("energy: $E")
end