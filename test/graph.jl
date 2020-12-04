using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

m = 6
n = 7
t = 4
g = Chimera(m, n, t)
@testset "Chimera creation" begin
   @test nv(g) == 2m * n * t
   @test ne(g) == t^2 * m * n + m * (n -1) * t + (m - 1) * n * t
   @test g[m, n, 2, t] == 2m * n * t
   @show g[1, 1]
end

@testset "Chimera outer connections" begin
   @test length(outer_connections(g, 1, 1, 3, 3)) == 0
   @test all(outer_connections(g, 1, 1, 1, 2) .== outer_connections(g, 1, 2, 1, 1))

   println(outer_connections(g, 1, 1, 1, 2))
   println(typeof(g))

   edges = filter_edges(g.graph, :outer, (1, 2))
   println(collect(edges))
end

@testset "Chimera/factor graph" begin
   M = 4
   N = 4
   T = 4

   C = 2 * N * M * T

   instance = "$(@__DIR__)/instances/chimera_droplets/$(C)power/001.txt" 

   ig = ising_graph(instance, C)
   cg = Chimera((M, N, T), ig)

   #fg = factor_graph(cg)

   for v ∈ vertices(cg)
      vv = filter_edges(cg.graph, :outer, (v, v))
      println(collect(vv))
   end
end
