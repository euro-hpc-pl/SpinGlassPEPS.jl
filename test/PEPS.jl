@testset "PepsTensor correctly builds PEPS tensor" begin
m = 4
n = 4
t = 4

L = 2 * n * m * t
instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

ig = ising_graph(instance, L)
cg = Chimera((m, n, t), ig)
fg = factor_graph(cg)

order = :PE
decompose_edges!(fg, order)

@testset "decompose_edges! workds correctly" begin
    order = get_prop(fg, :tensors_order)
    println("order ->", order)

    for e ∈ edges(fg)
        dec = get_prop(fg, e, :decomposition)
        en = get_prop(fg, e, :energy)

        println(size(en))
        println(size(first(dec)), size(last(dec)))

        if order == :PE
            @test en ≈ prod(dec)
        else
            @test en ≈ prod(reverse(dec))
        end
        break
    end
end

#peps = PepsTensor(fg, 6)

end