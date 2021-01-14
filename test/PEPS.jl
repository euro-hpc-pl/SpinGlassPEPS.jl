
@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1

L = m * n * t

bond_dimensions = [2, 2, 8, 4, 2, 2, 8]

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

for (bd, e) in zip(bond_dimensions, edges(fg))
    pl, en, pr = get_prop(fg, e, :split)
    println(e)
    println(size(pl), "   ", size(en),  "   ", size(pr))
   #=
    display(en)
    println("-------------------")
    println("-------------------")
    display(pl)
    println("-------------------")
    println("-------------------")
    display(pr)
    println("-------------------")
    println("-------------------")
    println("-------------------")
    =#
    isOK = min(size(en)...) == bd
    
    @test isOK
    if !isOK
        println(min(size(en)...), " ", bd)
        display(en)
        display(pl)
        display(pr)
    end
    
end


x, y = m, n

#for origin ∈ (:NW, :SW, :WN, :NE, :EN, :SE, :ES, :SW, :WS)
#for origin ∈ (:NW, :SW, :NE, :SE, :WN) # OK
for origin ∈ (:EN, ) #(:EN, :ES, :WS) # NO

    @info "testing peps" origin
    println(origin)

    peps = PepsNetwork(x, y, fg, β, origin)
    @test typeof(peps) == PepsNetwork

    for i ∈ 1:peps.i_max, j ∈ 1:peps.j_max
        A = SpinGlassPEPS._generate_tensor(peps, (i, j))
        B = generate_tensor(peps, (i, j))
        @test A ≈ B
    end

#=
    @info "contracting MPOs (up -> down)"

    ψ = MPO(PEPSRow(peps, 1))
    
    for A ∈ ψ @test size(A, 2) == 1 end

    for i ∈ 2:peps.i_max
        println(i)
        
        R = PEPSRow(peps, i)
        W = MPO(R)
        M = MPO(peps, i-1, i)

        println(ψ)
        println(M)
        println(W)

        ψ = (ψ * M) * W

        for A ∈ ψ @test size(A, 2) == 1 end
    end
=#
    #for A ∈ ψ @test size(A, 4) == 1 end

    @info "contracting MPOs (down -> up)"

    ψ = MPO(PEPSRow(peps, peps.i_max))

    for A ∈ ψ @test size(A, 4) == 1 end

    for i ∈ peps.i_max-1:1
        println(i)
        R = PEPSRow(peps, i)
        W = MPO(R) 
        M = MPO(peps, i, i+1) 

        println(W)
        println(M)
        println(ψ)

        ψ = W * (M * ψ)

        for A ∈ ψ @test size(A, 4) == 1 end
    end

    for A ∈ ψ @test size(A, 4) == 1 end

end
end
