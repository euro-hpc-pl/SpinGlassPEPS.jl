using MetaGraphs
using LightGraphs
using GraphPlot
using CSV


@testset "Lattice graph" begin
   m = 4
   n = 4
   t = 4

   L = 2 * n * m * t
   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   lt = Lattice((m, 1, n, 1, t), ig)

   @time fg = factor_graph(lt)

   @test collect(vertices(fg)) == collect(1:m * n)
   @test nv(fg) == m * n

   @info "Verifying cluster properties for Lattice" m, n, t

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

   @test isempty(intersect(clv...))
   @test isempty(intersect(cle...))
end

@testset "Rank reveal correctly decomposes energy row-wise" begin
   energy = [[1 2 3]; [0 -1 0]; [1 2 3]]
   P, E = rank_reveal(energy, :PE)
   @test size(P) == (3, 2)
   @test size(E) == (2, 3)
   @test P * E ≈ energy
end

@testset "Rank reveal correctly decomposes energy column-wise" begin
   energy = [[1, 2, 3] [0, -1, 1] [1, 2, 3]]
   E, P = rank_reveal(energy, :EP)
   @test size(P) == (2, 3)
   @test size(E) == (3, 2)
   @test E * P ≈ energy
end