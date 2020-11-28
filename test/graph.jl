using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

M = 2
N = 2
T = 4

C = 2 * N * M * T

instance = "$(@__DIR__)/instances/chimera_droplets/$(N)power/001.txt"  

@testset "Chimera graph" begin
    ig = ising_graph(instance, N)
    chimera = Chimera((M, N, T), ig)

    
end
