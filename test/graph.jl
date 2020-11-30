using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Chimera creation" begin
    m = 6
    n = 7
    t = 4
    g = Chimera(m, n, t)
    @test nv(g) == 2m * n * t
    @test ne(g) == t^2 * m * n + m * (n -1) * t + (m - 1) * n * t
    @test g[m, n, 2, t] == 2m * n * t
    @show g[1, 1]
end

 @testset "Chimera graph" begin
     M = 4
     N = 4
     T = 4

     C = 2 * N * M * T

     instance = "$(@__DIR__)/instances/chimera_droplets/$(C)power/001.txt"  
     ig = ising_graph(instance, N)
     chimera = Chimera((M, N, T), ig)

 end
