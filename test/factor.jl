using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "split_into_clusters correctly assings vertices to clusters" begin
   rule = Dict(
      1 => 1,
      2 => 1,
      3 => 3,
      4 => 2,
      5 => 3,
      6 => 4
   )
   vertices = [1, 3, 4, 5]
   expected_result = Dict(
      1 => [1], 2 => [4], 3 => [3, 5], 4 => []
   )

   @test split_into_clusters(vertices, rule) == expected_result
end

@testset "Lattice graph" begin
   m = 4
   n = 4
   t = 4
   L = n * m * (2 * t)

   instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt"

   ig = ising_graph(instance, L)

   fg = factor_graph(
      ig, 2, cluster_assignment_rule=chimera_to_square_lattice((m, n, 2*t))
   )

   @test collect(vertices(fg)) == collect(1:m * n)

   @info "Verifying cluster properties for Lattice" m, n, t

   clv = []
   cle = []
   rank = rank_vec(ig)

   for v ∈ vertices(fg)
      cl = get_prop(fg, v, :cluster)

      vmap = get_prop(cl, :vmap)
      push!(clv, vmap)
      push!(cle, collect(edges(cl)))

      for (i, v) in enumerate(vmap)
         @test rank_vec(cl)[i] == rank[v]
      end
   end

   # Check if graph is factored correctly
   @test isempty(intersect(clv...))
#   @test isempty(intersect(cle...))
end

@testset "Factor graph builds on pathological instance" begin
m = 3
n = 4
t = 3
L = n * m * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

ising = CSV.File(instance, types=[Int, Int, Float64], header=0, comment = "#")

couplings = Dict()
for (i, j, v) ∈ ising
    push!(couplings, (i, j) => v)
end

cedges = Dict(
   (1, 2) => [(1, 4), (1, 5), (1, 6)],
   (1, 5) => [(1, 13)],
   (2, 3) => [(4, 7), (5, 7), (6, 8), (6, 9)],
   (2, 6) => [(6, 16), (6, 18), (5, 16)],
   (5, 6) => [(13, 16), (13, 18)],
   (6, 10) => [(18, 28)],
   (10, 11) => [(28, 31), (28, 32), (28, 33), (29, 31), (29, 32), (29, 33), (30, 31), (30, 32), (30, 33)]
)

cells = Dict(
   1 => [1],
   2 => [4, 5, 6],
   3 => [7, 8, 9],
   4 => [],
   5 => [13],
   6 => [16, 18],
   7 => [],
   8 => [],
   9 => [],
   10 => [28, 29, 30],
   11 => [31, 32, 33],
   12 => []
)

d = 2
rank = Dict(
   c => fill(d, length(idx))
   for (c,idx) ∈ cells if !isempty(idx)
)

bond_dimensions = [2, 2, 8, 4, 2, 2, 8]

ig = ising_graph(instance)


fg = factor_graph(
    ig,
    energy=energy,
    spectrum=full_spectrum,
    cluster_assignment_rule=chimera_to_square_lattice((m, n, t)),
)

for v ∈ vertices(fg)
   cl = get_prop(fg, v, :cluster)
   @test sort(collect(nodes(cl))) == cells[v]
end


for (bd, e) in zip(bond_dimensions, edges(fg))
   pl, en, pr = get_prop(fg, e, :pl), get_prop(fg, e, :en), get_prop(fg, e, :pr)
   @test minimum(size(en)) == bd
end

for ((i, j), cedge) ∈ cedges
   pl, en, pr = get_prop(fg, i, j, :pl), get_prop(fg, i, j, :en), get_prop(fg, i, j, :pr)

   base_i = all_states(rank[i])
   base_j = all_states(rank[j])

   idx_i = enum(cells[i])
   idx_j = enum(cells[j])

   # Change it to test if energy is calculated using passed 'energy' function
   energy = zeros(prod(rank[i]), prod(rank[j]))

   for (ii, σ) ∈ enumerate(base_i)
      for (jj, η) ∈ enumerate(base_j)
         eij = 0.
         for (k, l) ∈ values(cedge)
            kk, ll = enum(cells[i])[k], enum(cells[j])[l]
            s, r = σ[idx_i[k]], η[idx_j[l]]
            J = couplings[k, l]
            eij += s * J * r
         end
         energy[ii, jj] = eij
      end
   end
   @test energy ≈ pl * (en * pr)
end

@testset "each cluster comprises expected cells" begin
for v ∈ vertices(fg)
   cl = get_prop(fg, v, :cluster)

   @test issetequal(nodes(cl), cells[v])
end
end

@testset "each edge comprises expected bunch of edges from source Ising graph" begin
for e ∈ edges(fg)
   outer_edges = get_prop(fg, e, :outer_edges)
   # println(collect(outer_edges))
   # println(cedges[Tuple(e)])
   # Note: this test is ok if we translate edges correctly.
   # TODO: fix this by translating from nodes to graph coordinates
   # @test issetequal(cedges[Tuple(e)], collect(outer_edges))
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
   @test Pl * E_old ≈ energy

   E, Pr = rank_reveal(E_old, :EP)
   @test size(Pr) == (8, 8)
   @test size(E) == (8, 8)
   @test E * Pr ≈ E_old
   @test Pl * E * Pr ≈ energy
end
