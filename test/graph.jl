using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

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

@testset "Lattice graph" begin
   m = 4
   n = 4
   t = 4

   β = 1

   L = n * m * (2 * t)
   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

   ig = ising_graph(instance, L)
   update_cells!(
      ig, 
      rule = square_lattice((m, n, 2*t)),
   ) 

   @time fg = factor_graph(ig)

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

   peps = PepsNetwork(m, n, fg, β, :NW)
   for i ∈ 1:m, j ∈ 1:n
      @time A = generate_tensor(peps, (i, j))
  end
end

@testset "Testing factor graph" begin
m = 3
n = 4
t = 3

β = 1
L = n * m * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt" 

edges = Dict()
push!(edges, (1, 2) => [(1, 4), (1, 5), (1, 6)])
push!(edges, (1, 5) => [(1, 13)])

push!(edges, (2, 3) => [(4, 7), (5, 7), (6, 8), (6, 9)])
push!(edges, (2, 6) => [(6, 16), (6, 18), (5, 16)])

push!(edges, (5, 6) => [(13, 16), (13, 18)])

push!(edges, (6, 10) => [(18, 28)])
push!(edges, (10, 11) => [(28, 31), (28, 32), (28, 33), (29, 31), (29, 32), (29, 33), (30, 31), (30, 32), (30, 33)])

cells = Dict()
push!(cells, 1 => [1])
push!(cells, 2 => [4, 5, 6])
push!(cells, 3 => [7, 8, 9])
push!(cells, 4 => [])
push!(cells, 5 => [13])
push!(cells, 6 => [16, 18])
push!(cells, 7 => [])
push!(cells, 8 => [])
push!(cells, 9 => [])
push!(cells, 10 => [28, 29, 30])
push!(cells, 11 => [31, 32, 33])
push!(cells, 12 => [])

ig = ising_graph(instance, L)
update_cells!(
   ig, 
   rule = square_lattice((m, n, t)),
) 

fg = factor_graph(
    ig, 
    energy=energy, 
    spectrum=full_spectrum,
)

for v ∈ vertices(fg)
   cl = get_prop(fg, v, :cluster)
  
   @test issetequal(keys(cl.vertices), cells[v])

   for w ∈ neighbors(fg, v)
      ed = get_prop(fg, v, w, :edge)
      for e in edges[(v, w)] @test e ∈ Tuple.(ed.edges) end
      for e in ed.edges @test Tuple(e) ∈ edges[(v, w)] end
   end
end

end