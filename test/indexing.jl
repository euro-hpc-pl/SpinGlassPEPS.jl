
@testset "Factor graph correctly indexing states" begin

#     A1    |  A2
#           |
#   1 -- 2 -|- 3

instance = Dict{Tuple{Int64,Int64},Float64}()

push!(instance, (1,1) => 0.704)
push!(instance, (2,2) => 0.868)
push!(instance, (3,3) => 0.592)

push!(instance, (1, 2) => 0.652)
push!(instance, (2, 3) => 0.730)

m = 1
n = 2
t = 2

β = 1

L = m * n * t

ig = ising_graph(instance, L)

fg = factor_graph(
    ig,
    Dict(1=>4, 2=>1),
    energy=energy,
    spectrum=full_spectrum,
)

fg_bf = factor_graph(
    ig,
    Dict(1=>4, 2=>1),
    energy=energy,
    spectrum=brute_force,
)

sp = get_prop(fg, 1, :spectrum)
sp_bf = get_prop(fg_bf, 1, :spectrum)

display(sp.states)
display(sp_bf.states)

end