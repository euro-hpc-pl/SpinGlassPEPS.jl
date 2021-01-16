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
   t = 4

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
end

@testset "Factor graph builds on pathological instance" begin
m = 3
n = 4
t = 3

β = 1
L = n * m * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

ising = CSV.File(instance, types=[Int, Int, Float64], header=0, comment = "#")

couplings = Dict()
for (i, j, v) ∈ ising
    push!(couplings, (i, j) => v)
end

cedges = Dict()
push!(cedges, (1, 2) => [(1, 4), (1, 5), (1, 6)])
push!(cedges, (1, 5) => [(1, 13)])

push!(cedges, (2, 3) => [(4, 7), (5, 7), (6, 8), (6, 9)])
push!(cedges, (2, 6) => [(6, 16), (6, 18), (5, 16)])

push!(cedges, (5, 6) => [(13, 16), (13, 18)])

push!(cedges, (6, 10) => [(18, 28)])
push!(cedges, (10, 11) => [(28, 31), (28, 32), (28, 33), (29, 31), (29, 32), (29, 33), (30, 31), (30, 32), (30, 33)])

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

d = 2
rank = Dict()
for (c, idx) ∈ cells
   if !isempty(idx) 
      push!(rank, c => fill(d, length(idx)))
   end
end

bond_dimensions = [2, 2, 8, 4, 2, 2, 8]

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


for (bd, e) in zip(bond_dimensions, edges(fg))
   pl, en, pr = get_prop(fg, e, :split)
   display(e)
   println(size(pl), "   ", size(en),  "   ", size(pr))

   @test min(size(en)...) == bd
end

for (i, j) ∈ keys(cedges)
   pl, en, pr = get_prop(fg, i, j, :split)
   
   base_i = all_states(rank[i])
   base_j = all_states(rank[j])

   idx_i = enum(cells[i])
   idx_j = enum(cells[j])

   energy = zeros(prod(rank[i]), prod(rank[j]))

   for (ii, σ) ∈ enumerate(base_i)
      for (jj, η) ∈ enumerate(base_j)
         eij = 0.
         for (k, l) ∈ values(cedges[i, j])
            kk, ll = enum(cells[i])[k], enum(cells[j])[l]
            s, r = σ[idx_i[k]], η[idx_j[l]]
            J = couplings[k, l]
            eij += s * J * r 
         end
         energy[ii, jj] = eij
      end
   end

   println("Edge ", i, " => ", j)
   display(energy)

   @test energy ≈ pl * (en * pr)
end

for v ∈ vertices(fg)
   cl = get_prop(fg, v, :cluster)
  
   @test issetequal(keys(cl.vertices), cells[v])

   for w ∈ neighbors(fg, v)
      ed = get_prop(fg, v, w, :edge)
      for e in cedges[(v, w)] @test e ∈ Tuple.(ed.edges) end
      for e in ed.edges @test Tuple(e) ∈ cedges[(v, w)] end
   end
end

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

@testset "Rank reveal correctly decomposes energy into projector, energy, projector" begin
   #energy = [[1 2 3]; [0 -1 0]; [1 2 3]]
   energy = [[1.0  -0.5   1.5   0.0   0.0  -1.5   0.5  -1.0];
   [0.0   0.5   0.5   1.0  -1.0  -0.5  -0.5   0.0];
   [0.5   0.0   1.0   0.5  -0.5  -1.0   0.0  -0.5];
   [-0.5   1.0   0.0   1.5  -1.5   0.0  -1.0   0.5];
   [0.5  -1.0   0.0  -1.5   1.5   0.0   1.0  -0.5];
   [-0.5   0.0  -1.0  -0.5   0.5   1.0   0.0   0.5];
   [0.0  -0.5  -0.5  -1.0   1.0   0.5   0.5   0.0];
   [-1.0   0.5  -1.5   0.0   0.0   1.5  -0.5   1.0]]
   Pl, E_old = rank_reveal(energy, :PE)
   @test size(Pl) == (8, 8)
   @test size(E_old) == (8, 8)
   println("energy: ")
   display(energy)
   println("Pl: ")
   display(Pl)
   println("E_old: ")
   display(E_old)
   @test Pl * E_old ≈ energy

   E, Pr = rank_reveal(E_old, :EP)
   @test size(Pr) == (8, 8)
   @test size(E) == (8, 8)
   println("E: ")
   display(E)
   println("Pr: ")
   display(Pr)
   @test E * Pr ≈ E_old
   @test Pl * E * Pr ≈ energy
end