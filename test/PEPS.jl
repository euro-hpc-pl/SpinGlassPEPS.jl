@testset "PepsTensor correctly builds PEPS tensor" begin
m = 4
n = 4
t = 4

L = 2 * n * m * t
instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

ig = ising_graph(instance, L)
cg = Chimera((m, n, t), ig)

β = get_prop(ig, :β)

for order ∈ (:EP, :PE)
    for hd ∈ (:LR, :RL), vd ∈ (:BT, :TB)

        @info "Testing factor graph" order hd vd

        fg = factor_graph(cg, hdir=hd, vdir=vd)
        decompose_edges!(fg, order, β=β)

        @test order == get_prop(fg, :tensors_order)
        @test (hd, vd) == get_prop(fg, :order)

        for e ∈ edges(fg)
            dec = get_prop(fg, e, :decomposition)
            en = get_prop(fg, e, :energy)

            @test exp.(-β .* en) ≈ prod(dec)
        end

        @info "Testing PEPS"
        
        @time begin
            net = []
            for v ∈ vertices(fg)
                peps = PepsTensor(fg, v)
                push!(net, peps)
                println(size(peps))
            end
        end    
    end
end

end