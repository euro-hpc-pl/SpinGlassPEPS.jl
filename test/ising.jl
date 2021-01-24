using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Ising" begin

    L = 4
    N = L^2 
    instance = "$(@__DIR__)/instances/$(N)_001.txt"  

    ig = ising_graph(instance, N)

    E = get_prop(ig, :energy)

    println(ig)
    println("energy: $E")

    for spin ∈ vertices(ig)
        println("neighbors of spin $spin are: ", neighbors(ig, spin) )
    end

    @test nv(ig) == N

    for i ∈ 1:N
        @test has_vertex(ig, i)
    end    

    A = adjacency_matrix(ig)
    display(Matrix{Int}(A))
    println("   ")

    B = zeros(Int, N, N)
    for i ∈ 1:N
        nbrs = unique_neighbors(ig, i)
        for j ∈ nbrs
            B[i, j] = 1
        end    
    end

    @test B + B' == A
   
    gplot(ig, nodelabel=1:N)

    @testset "Naive brute force for +/-1" begin
        k = 2^N

        sp = brute_force(ig, num_states=k)

        s = 5
        display(sp.states[1:s])
        println("   ")
        display(sp.energies[1:s])
        println("   ")

        @test sp.energies ≈ energy.(sp.states, Ref(ig))

        # states, energies = brute_force(ig, num_states=k)

        # @test energies ≈ sp.energies
        # @test states == sp.states

        β = rand(Float64)
        ρ = gibbs_tensor(ig, β)

        @test size(ρ) == Tuple(fill(2, N))

        r = exp.(-β .* sp.energies)
        R = r ./ sum(r)

        @test sum(R) ≈ 1
        @test sum(ρ) ≈ 1        

        @test [ ρ[idx.(σ)...] for σ ∈ sp.states ] ≈ R
    end

    @testset "Naive brute force for general spins" begin
        L = 4 
        instance = "$(@__DIR__)/instances/$(L)_001.txt"  

        ig = ising_graph(instance, L)

        set_prop!(ig, :rank, [3,2,5,4])
        rank = get_prop(ig, :rank)

        all = prod(rank)
        sp = brute_force(ig, num_states=all)

        β = rand(Float64)
        ρ = exp.(-β .* sp.energies)

        ϱ = ρ ./ sum(ρ) 
        ϱ̃ = gibbs_tensor(ig, β)
 
        @test [ ϱ̃[idx.(σ)...] for σ ∈ sp.states ] ≈ ϱ 
    end

    @testset "Reading from Dict" begin
        instance_dict = Dict()
        ising = CSV.File(instance, types=[Int, Int, Float64], header=0, comment = "#")

        for (i, j, v) ∈ ising
            push!(instance_dict, (i, j) => v)
        end

        ig = ising_graph(instance, N)
        ig_dict = ising_graph(instance_dict, N)

        @test gibbs_tensor(ig) ≈ gibbs_tensor(ig_dict)
    end 
end

@testset "Ground state energy for pathological instance " begin
m = 3
n = 4
t = 3

β = 1
L = n * m * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

ising = CSV.File(instance, types=[Int, Int, Float64], header=0, comment = "#")

couplings = Dict()
for (i, j, v) ∈ ising
    push!(couplings, (i, j) => v)
end

cedges = Dict()
push!(cedges, (1, 2) => [(1, 4), (1, 5), (1, 6)])
push!(cedges, (1, 5) => [(1, 13)])

push!(cedges, (2, 3) => [(4, 7), (5, 7), (6, 8), (6, 9)])
push!(cedges, (2, 6) => [(6, 16), (6, 18), (5, 16)])

push!(cedges, (5, 6) => [(13, 16), (13, 18)])

push!(cedges, (6, 10) => [(18, 28)])
push!(cedges, (10, 11) => [(28, 31), (28, 32), (28, 33), (29, 31), (29, 32), (29, 33), (30, 31), (30, 32), (30, 33)])

push!(cedges, (2, 2) => [(4, 5), (4, 6), (5, 6), (6, 6)])
push!(cedges, (3, 3) => [(7, 8), (7, 9)])
push!(cedges, (6, 6) => [(16, 18), (16, 16)])
push!(cedges, (10, 10) => [(28, 29), (28, 30), (29, 30)])

cells = Dict()
push!(cells, 1 => [1])
push!(cells, 2 => [4, 5, 6])
push!(cells, 3 => [7, 8, 9])
push!(cells, 4 => [])
push!(cells, 5 => [13])
push!(cells, 6 => [16, 18])
push!(cells, 7 => [])
push!(cells, 8 => [])
push!(cells, 9 => [])
push!(cells, 10 => [28, 29, 30])
push!(cells, 11 => [31, 32, 33])
push!(cells, 12 => [])

d = 2
rank = Dict()
for (c, idx) ∈ cells
   if !isempty(idx) 
      push!(rank, c => fill(d, length(idx)))
   end
end

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

low_energies = [-16.4, -16.4, -16.4, -16.4, -16.1, -16.1, -16.1, -16.1, -15.9, -15.9, -15.9, -15.9, -15.9, -15.9, -15.6, -15.6, -15.6, -15.6, -15.6, -15.6, -15.4, -15.4]
configurations = Dict()
push!(configurations, 1 => [-1, -1, -1, -1])
push!(configurations, 2 => [0, 0, 0, 0])
push!(configurations, 3 => [0, 0, 0, 0])
push!(configurations, 4 => [1, 1, 1, 1])
push!(configurations, 5 => [1, 1, 1, 1])
push!(configurations, 6 => [-1, -1, -1, -1])
push!(configurations, 7 => [-1, -1, -1, -1])
push!(configurations, 8 => [-1, -1, 1, 1])
push!(configurations, 9 => [1, 1, 1, 1])
push!(configurations, 10 => [0, 0, 0, 0])
push!(configurations, 11 => [0, 0, 0, 0])
push!(configurations, 12 => [0, 0, 0, 0])
push!(configurations, 13 => [1, 1, 1, 1])
push!(configurations, 14 => [0, 0, 0, 0])
push!(configurations, 15 => [0, 0, 0, 0])
push!(configurations, 16 => [1, 1, 1, 1])
push!(configurations, 17 => [0, 0, 0, 0])
push!(configurations, 18 => [-1, -1, -1, -1])
push!(configurations, 19 => [0, 0, 0, 0])
push!(configurations, 20 => [0, 0, 0, 0])
push!(configurations, 21 => [0, 0, 0, 0])
push!(configurations, 22 => [0, 0, 0, 0])
push!(configurations, 23 => [0, 0, 0, 0])
push!(configurations, 24 => [0, 0, 0, 0])
push!(configurations, 25 => [0, 0, 0, 0])
push!(configurations, 26 => [0, 0, 0, 0])
push!(configurations, 27 => [0, 0, 0, 0])
push!(configurations, 28 => [1, 1, 1, 1])
push!(configurations, 29 => [1, 1, 1, 1])
push!(configurations, 30 => [-1, -1, -1, -1])
push!(configurations, 31 => [1, 1, 1, 1])
push!(configurations, 32 => [-1, -1, -1, -1])
push!(configurations, 33 => [1,-1, 1, -1])
push!(configurations, 34 => [0, 0, 0, 0])
push!(configurations, 35 => [0, 0, 0, 0])
push!(configurations, 36 => [0, 0, 0, 0])

e = zeros(1,4)
for (i, j) ∈ keys(cedges)
    for (k, l) ∈ values(cedges[i, j])
        for m ∈ 1:length(configurations[k])
            s = configurations[k][m]
            r = configurations[l][m]
            J = couplings[k, l]
            if k == l
                e[m] += dot(s,J)
            else
                e[m] += dot(s, J, r)
            end
        end
   end
end
@test e[1] == e[2] == e[3] == e[4]
@test e[1] == low_energies[1]
@test e[2] == low_energies[2]
@test e[3] == low_energies[3]
@test e[4] == low_energies[4]
println("low energies: ", e)
end