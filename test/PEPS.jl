
@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1

L = m * n * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt" 

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

decompose_edges!(fg)

x = m
y = n
peps = PepsNetwork(x, y, fg, β)

#=
for i ∈ 1:x, j ∈ 1:y
    @time A = generate_tensor(peps, (i, j))
end
=#
end