
@testset "Custom settings work with factor graph" begin

#     A1    |  A2
#           |
#   1 -- 2 -|- 3

instance = Dict{Tuple{Int64, Int64}, Float64}()

push!(instance, (1,1) => 0.704)
push!(instance, (2,2) => 0.868)
push!(instance, (3,3) => 0.592)

push!(instance, (1, 2) => 0.652)
push!(instance, (2, 3) => 0.730)

custom_rule = Dict(1 => 1, 2 => 1, 3 => 2)
custom_spec = Dict(1 => 3, 2 => 1)

ig = ising_graph(instance, 3)
update_cells!(
   ig,
   rule = custom_rule,
)

fg = factor_graph(
    ig,
    custom_spec,
    energy=energy,
    spectrum=full_spectrum,
)

for v in vertices(fg)
    cl = get_prop(fg, v, :cluster)
    sp = get_prop(fg, v, :spectrum)

    @test length(sp.energies) == length(sp.states) == custom_spec[v]
    @test collect(keys(cl.vertices)) == cluster(v, custom_rule)
end

end