@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1

L = m * n * t

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

x, y = m, n

#for origin ∈ (:NW, :SW, :WN, :NE, :EN, :SE, :ES, :SW, :WS)
for origin ∈ (:NW, :SW, :NE, :SE, :SW)

    @info "testing peps" origin
    println(origin)

    peps = PepsNetwork(x, y, fg, β, origin)
    @test typeof(peps) == PepsNetwork

    for i ∈ 1:peps.i_max, j ∈ 1:peps.j_max
        A = SpinGlassPEPS._generate_tensor(peps, (i, j)) 
        B = generate_tensor(peps, (i, j))
        @test A ≈ B
    end

    @info "contracting MPOs"

    mpo = MPO(peps, 1)
    for ψ ∈ mpo
        println(size(ψ))
        @test size(ψ, 2) == 1
    end

    for i ∈ 2:peps.i_max
        mpo *= MPO(peps, i) 
    end

    for ψ ∈ mpo
        #println(size(ψ))
        @test size(ψ, 2) == 1
        @test size(ψ, 4) == 1
    end
end

end