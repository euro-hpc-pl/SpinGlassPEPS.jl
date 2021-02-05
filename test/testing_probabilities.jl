import SpinGlassPEPS: Partial_sol, update_partial_solution, M2graph
import SpinGlassPEPS: energy, solve, contract

Mq = zeros(9,9)
Mq[1,1] = 1.
Mq[2,2] = 1.4
Mq[3,3] = -0.2
Mq[4,4] = -1.
Mq[5,5] = 0.2
Mq[6,6] = -2.2
Mq[7,7] = 0.2
Mq[8,8] = -0.2
Mq[9,9] = -0.8
Mq[1,2] = Mq[2,1] = 2.
Mq[1,4] = Mq[4,1] = -1.
Mq[2,3] = Mq[3,2] = 1.1
Mq[4,5] = Mq[5,4] = 0.5
Mq[4,7] = Mq[7,4] = -1.
Mq[2,5] = Mq[5,2] = -.75
Mq[3,6] = Mq[6,3] = 1.5
Mq[5,6] = Mq[6,5] = 2.1
Mq[5,8] = Mq[8,5] = 0.12
Mq[6,9] = Mq[9,6] = -0.52
Mq[7,8] = Mq[8,7] = 0.5
Mq[8,9] = Mq[9,8] = -0.05

@testset "testing marginal/conditional probabilities" begin

    ####   conditional probability implementation


    β = 3.
    g = M2graph(Mq, -1)

    bf = brute_force(g; num_states = 1)

    rule = Dict{Any,Any}(1 => 1, 2 => 1, 4 => 1, 5 => 1, 3=>2, 6 => 2, 7 => 3, 8 => 3, 9 => 4)

    update_cells!(
      g,
      rule = rule,
    )

    fg = factor_graph(
        g,
        energy=energy,
        spectrum=brute_force,
    )

    origin = :NW
    states = collect.(all_states(rank_vec(g)))
    ρ = exp.(-β .* energy.(states, Ref(g)))
    Z = sum(ρ)

    println("grid of nodes")
    display([1 2 ; 3 4])
    println()

    peps = PepsNetwork(2, 2, fg, β, origin)
    Dcut = 8
    tol = 0.
    sweep = 4
    z = contract(peps, Dict{Int, Int}())
    @test z ≈ Z

    boundary_mps = boundaryMPS(peps, Dcut, tol, sweep)

    # initialize
    ps = Partial_sol{Float64}(Int[], 0.)

    #node 1
    obj1 = conditional_probabs(peps, ps, boundary_mps[1], PEPSRow(peps, 1))
    _, i = findmax(obj1)
    ps1 = update_partial_solution(ps, i, obj1[i])

    # test all
    for i_1 in 1:16
        @test obj1[i_1] ≈ contract(peps, Dict{Int, Int}(1 => i_1))/z atol = 1e-3
    end
    # test chosen
    @test (props(fg, 1)[:spectrum]).states[i] == [bf.states[1][a] for a in [1,2,4,5]]

    # node 2
    obj2 = conditional_probabs(peps, ps1, boundary_mps[1], PEPSRow(peps, 1))
    obj2 = obj1[i].*obj2
    _, j = findmax(obj2)
    ps2 = update_partial_solution(ps1, j, obj2[j])


    @test (props(fg, 2)[:spectrum]).states[j] == [bf.states[1][a] for a in [3,6]]

    for i_2 in 1:4
        @test obj2[i_2] ≈ contract(peps, Dict{Int, Int}(1 => i, 2 =>i_2))/z atol = 1e-3
    end

    println(" .............  node 3 ......")
    println("tensor")
    display(reshape(PEPSRow(peps, 2)[1], (4,2,4)))
    println()
    println("done from 2 spins, 2 bounds up and 1 left")
    println("shoud be constructed of one couplung matrix (up) and projector (left)")
    println("only 4 non-zero elements suggests 2 projectors or lacking elements")
    println()

    # node 3
    obj3 = conditional_probabs(peps, ps2, boundary_mps[2], PEPSRow(peps, 2))
    obj3 = obj2[j].*obj3

    _, k = findmax(obj3)
    ps3 = update_partial_solution(ps2, k, obj3[k])

    for i_3 in 1:4
        @test obj3[i_3] ≈ contract(peps, Dict{Int, Int}(1 => i, 2 =>j, 3 => i_3))/z atol = 1e-3
    end

    @test (props(fg, 3)[:spectrum]).states[k] == [bf.states[1][a] for a in [7,8]]

    #node 4

    obj4 = conditional_probabs(peps, ps3, boundary_mps[2], PEPSRow(peps, 2))
    obj4 = obj3[k].*obj4

    _, l = findmax(obj4)

    for i_4 in 1:2
        @test obj4[i_4] ≈ contract(peps, Dict{Int, Int}(1 => i, 2 =>j, 3 => k, 4 => i_4))/z atol = 1e-3
    end

    @test (props(fg, 4)[:spectrum]).states[l] == [bf.states[1][a] for a in [9]]

    println(props(fg, 4)[:spectrum])

    println(" .............. node 4 .............")
    println("tensor should have no non-zero elements, should be composed of two coupling matrices")
    display(reshape(PEPSRow(peps, 2)[2], (2,2,2)))
end
