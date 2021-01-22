

@testset "Factor graph correctly indexing states" begin

m = 3
n = 3
t = 1

β = 1

L = m * n * t

instance = "$(@__DIR__)/instances/$(L)_001.txt"

ig = ising_graph(instance, L)

fg = factor_graph(
    ig,
    energy=energy,
    spectrum=full_spectrum,
)

fg_bf = factor_graph(
    ig,
    energy=energy,
    spectrum= x -> brute_force(x, num_states=2),
)

sp = get_prop(fg, 1, :spectrum)
sp_bf = get_prop(fg_bf, 1, :spectrum)

display(sp.states)
display(sp_bf.states)

end