@testset "PepsTensor correctly builds PEPS tensor" begin
m = 4
n = 4
t = 4

L = 2 * n * m * t
instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

ig = ising_graph(instance, L)
cg = Chimera((m, n, t), ig)

β = get_prop(ig, :β)

order = :PE
fg = factor_graph(cg)
decompose_edges!(fg, order, β=β)

@testset "decompose_edges! workds correctly" begin
    order = get_prop(fg, :tensors_order)

    for e ∈ edges(fg)
        dec = get_prop(fg, e, :decomposition)
        en = get_prop(fg, e, :energy)

        ρ = exp.(-β .* en)
        if order == :PE
            @test ρ ≈ prod(dec)
        else
            @test ρ ≈ prod(reverse(dec))
        end
        break
    end
end

#peps = PepsTensor(fg, 6)

end