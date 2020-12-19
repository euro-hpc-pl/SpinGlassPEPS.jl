@testset "PepsTensor correctly builds PEPS tensor" begin
m = 4
n = 4
t = 4

L = 2 * n * m * t
instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

ig = ising_graph(instance, L)
cg = Chimera((m, n, t), ig)

fg = factor_graph(cg)
decompose_edges!(fg)

#peps = PepsTensor(fg, 6)

end