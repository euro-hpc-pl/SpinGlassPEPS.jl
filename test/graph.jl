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

   # This still does not work
   println(outer_connections(g, 1, 1, 1, 2))
   println(typeof(g))

   edges = filter_edges(g.graph, :cluster, (1, 2))
   println(collect(edges))
end

@testset "Chimera/factor graph" begin
   m = 16
   n = 16
   t = 4

   L = 2 * n * m * t

   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   cg = Chimera((m, n, t), ig)

   cl = cluster(cg, 1)
   rank = get_prop(cg.graph, :rank)
   sp = all_states(rank[collect(cl.vertices)])

   @time en = energy.(sp, Ref(cg.graph), Ref(cl))

   #@test en ≈ en2
   #@time fg = factor_graph(cg)
end
