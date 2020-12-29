
@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

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

#=
ng = NetworkGraph(fg,)

for v ∈ vertices(fg)
    A = generate_tensor(ng, v)

    for w ∈ neighbors(v)
        A = generate_tensor(ng, v, w)
    end
end
=#

end