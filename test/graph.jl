using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

m = 6
n = 7
t = 4
@time g = Chimera(m, n, t)
@testset "Chimera creation" begin
   @test nv(g) == 2m * n * t
   @test ne(g) == t^2 * m * n + m * (n -1) * t + (m - 1) * n * t
   @test g[m, n, 2, t] == 2m * n * t
   @show g[1, 1]
end

@testset "Chimera graph" begin
   m = 4
   n = 4
   t = 4

   L = 2 * n * m * t
   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   cg = Chimera((m, n, t), ig)

   @time fg = factor_graph(cg)

   @test collect(vertices(fg)) == collect(1:m * n)
   @test nv(fg) == m * n

   @info "Verifying cluster properties for Chimera" m, n, t

   clv = []
   cle = []
   rank = get_prop(ig, :rank)

   for v ∈ vertices(fg)
      cl = get_prop(fg, v, :cluster)
      push!(clv, keys(cl.vertices))
      push!(cle, collect(cl.edges))

      for (g, l) ∈ cl.vertices
         @test cl.rank[l] == rank[g]
      end
   end

   println(intersect(clv)...)
   #@test isempty(intersect(cle))
end

@testset "Factor graph" begin
   m = 16
   n = 16
   t = 4

   L = 2 * n * m * t
   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   cg = Chimera((m, n, t), ig)

   @time fg = factor_graph(cg)
end

@testset "Rank reveal correctly decomposes energy row-wise" begin
   energy = [[1 2 3]; [0 -1 0]; [1 2 3]]
   P, E = rank_reveal(energy, P_then_E)
   @test size(P) == (3, 2)
   @test size(E) == (2, 3)
   @test P * E ≈ energy
end

@testset "Rank reveal correctly decomposes energy column-wise" begin
   energy = [[1, 2, 3] [0, -1, 1] [1, 2, 3]]
   E, P = rank_reveal(energy, E_then_P)
   @test size(P) == (2, 3)
   @test size(E) == (3, 2)
   @test E * P ≈ energy
end