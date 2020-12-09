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


@testset "Chimera/factor graph" begin
   m = 4
   n = 4
   t = 4

   L = 2 * n * m * t

   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   cg = Chimera((m, n, t), ig)

   #cl = Cluster(cg, 2)
   #@time Spectrum(cl)

   @time fg = factor_graph(cg)
end
