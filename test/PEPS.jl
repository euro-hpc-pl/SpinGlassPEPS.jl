
@testset "PepsTensor correctly builds PEPS network for Lattice" begin

L = 3
N = L^2
instance = "$(@__DIR__)/instances/$(N)_001.txt" 

ig = ising_graph(instance, N)
fg = factor_graph(ig, energy=energy, spectrum=full_spectrum)

#=
decompose_edges!(fg)
ng = NetworkGraph(fg,)

for v ∈ vertices(fg)
    peps = PepsTensor(fg, v)  
end
=#
end