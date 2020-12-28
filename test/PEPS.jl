@testset "PepsTensor correctly builds PEPS network for Chimera" begin
m = 4
n = 4
t = 4

L = 2 * n * m * t
instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt" 

ig = ising_graph(instance, L)
cg = Chimera((m, n, t), ig)

β = get_prop(ig, :β)
k = 64

#=
for order ∈ (:EP, :PE)
    for hd ∈ (:LR, :RL), vd ∈ (:BT, :TB)

        @info "Testing factor graph" order hd vd

        fg = factor_graph(cg,
        #spectrum=full_spectrum,
        spectrum = cl -> brute_force(cl, num_states=k),
        energy=energy,
        cluster=unit_cell,  
        hdir=hd,
        vdir=vd,
        )
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
                @test v == peps.tag
                push!(net, peps)
                println(peps.nbrs)
                println(size(peps))
            end
        end    
    end
end
=#

@testset "PepsTensor correctly builds PEPS network for Lattice" begin

L = 3
N = L^2
instance = "$(@__DIR__)/instances/$(N)_001.txt" 

ig = ising_graph(instance, N)
lt = Lattice((L, L), ig)

for order ∈ (:EP, :PE)
    for hd ∈ (:LR, :RL), vd ∈ (:BT, :TB)

        @info "Testing factor graph" order hd vd

        fg = factor_graph(lt,
        spectrum=full_spectrum,
        energy=energy,
        cluster=unit_cell,  
        hdir=hd,
        vdir=vd,
        )

        decompose_edges!(fg, order, β=β)

        #ψ = MPO(fg, :r, 1)

        #@time begin
            net = []
            for v ∈ vertices(fg)
                peps = PepsTensor(fg, v)
                @test v == peps.tag
                push!(net, peps)
                @info "lt peps" peps.nbrs size(peps)
            end
        #end   

    end
end

end


end