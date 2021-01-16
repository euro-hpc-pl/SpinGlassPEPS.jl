
@testset "LinearIndices correctly assigns indices" begin
m = 3
n = 4

origin_l = [:NW, :NE, :SE, :SW]
origin_r = [:WN, :EN, :ES, :WS]

for (ol, or) ∈ zip(origin_l, origin_r)   
    ind_l, i_max_l, j_max_l = LinearIndices(m, n, ol)
    ind_r, i_max_r, j_max_r = LinearIndices(m, n, or)

    @test i_max_l == m == j_max_r
    @test j_max_l == n == i_max_r

    for i ∈ 0:m+1, j ∈ 0:n+1
        @test ind_l[i, j] == ind_r[j, i]
    end
end
end

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

#for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
for origin ∈ (:NW, :SW, :WS, :WN, :NE, :SE)
#
#for origin ∈ (:ES, :EN) # NO
#for origin ∈ (:WS, :WN) # NO if en'

    @info "testing peps" origin

    peps = PepsNetwork(x, y, fg, β, origin)
    @test typeof(peps) == PepsNetwork

    @info "contracting MPOs (up -> down)"

    ψ = MPO(PEPSRow(peps, 1))
    
    for A ∈ ψ @test size(A, 2) == 1 end

    for i ∈ 2:peps.i_max
        println("row -> ", i)

        R = PEPSRow(peps, i)
        W = MPO(R)
        M = MPO(peps, i-1, i)

        println(ψ)
        println(M)
        println(W)
        
        ψ = (ψ * M) * W

        for A ∈ ψ @test size(A, 2) == 1 end
        
        @test size(ψ[1], 1) == 1
        @test size(ψ[peps.j_max], 3) == 1
    end

    #=
    for A ∈ ψ @test size(A, 4) == 1 end
    println(ψ)

    @info "contracting MPOs (down -> up)"

    ψ = MPO(PEPSRow(peps, peps.i_max))

    for A ∈ ψ @test size(A, 4) == 1 end
    
    for i ∈ peps.i_max-1:-1:1
        println("row -> ", i)

        R = PEPSRow(peps, i)
        W = MPO(R) 
        M = MPO(peps, i, i+1) 

        println(W)
        println(M)
        println(W)

        ψ = W * (M * ψ)

        for A ∈ ψ @test size(A, 4) == 1 end

        @test size(ψ[1], 1) == 1
        @test size(ψ[peps.j_max], 3) == 1
    end
    
    for A ∈ ψ @test size(A, 4) == 1 end
    println(ψ)
    =#
end
end 


#=
@testset "Partition function from PEPS network" begin

m = 3
n = 3
t = 1

β = 1

L = m * n * t

instance = "$(@__DIR__)/instances/$(L)_001.txt"

ig = ising_graph(instance, L)

states = collect.(all_states(get_prop(ig, :rank)))
ρ = exp.(-β .* energy.(states, Ref(ig)))
Z = sum(ρ)

fg = factor_graph(
    ig,
    energy=energy,
    spectrum=full_spectrum,
)

peps = PepsNetwork(m, n, fg, β)

ψ = MPO(PEPSRow(peps, 1))
for i ∈ 2:peps.i_max
    W = MPO(PEPSRow(peps, i))
    M = MPO(peps, i-1, i)
    ψ = (ψ * M) * W
end

end
=#
