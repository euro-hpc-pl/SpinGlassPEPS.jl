@testset "LinearIndices correctly assigns indices" begin
m = 3
n = 4

origin_l = [:NW, :NE, :SE, :SW]
origin_r = [:WN, :EN, :ES, :WS]

for (i, (ol, or)) ∈ enumerate(zip(origin_l, origin_r))    

    println("origin ", ol, " ", or)

    ind_l, i_max_l, j_max_l = LinearIndices(m, n, ol)
    ind_r, i_max_r, j_max_r = LinearIndices(m, n, or)

    @test i_max_l == m == j_max_r
    @test j_max_l == n == i_max_r

    for i ∈ 0:m+1, j ∈ 0:n+1
        @test ind_l[i,j] == ind_r[j,i]
    end
end
end

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

    @test min(size(en)...) == bd
end

x, y = m, n

for origin ∈ (:NW, :SW, :WN, :NE, :EN, :SE, :ES, :SW, :WS)

    @info "testing peps" origin
    println(origin)

    peps = PepsNetwork(x, y, fg, β, origin)
    @test typeof(peps) == PepsNetwork

    for i ∈ 1:peps.i_max, j ∈ 1:peps.j_max
        A = SpinGlassPEPS._generate_tensor(peps, (i, j)) 
        B = generate_tensor(peps, (i, j))
        @test A ≈ B
    end

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

    for A ∈ ψ @test size(A, 4) == 1 end

    @info "contracting MPOs (down -> up)"

    ψ = MPO(PEPSRow(peps, peps.i_max))

    for A ∈ ψ @test size(A, 4) == 1 end

    println("imax -> ", peps.i_max)
    
    for i ∈ peps.i_max-1:-1:1
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