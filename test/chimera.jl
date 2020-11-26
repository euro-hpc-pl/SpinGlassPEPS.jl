using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

C = 4
N = 8 * C^2 

instance = "$(@__DIR__)/instances/chimera_droplets/$(N)power/001.txt"  

@testset "Chimera graph" begin
    ig = ising_graph(instance, N)

    println( [ get_prop(ig, i, :h) for i ∈ 1:nv(ig) ] )
end
