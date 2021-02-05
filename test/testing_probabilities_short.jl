import SpinGlassPEPS: Partial_sol, update_partial_solution, M2graph
import SpinGlassPEPS: energy, solve, contract

Mq = zeros(4,4)
Mq[1,1] = 1.
Mq[2,2] = 1.4
Mq[3,3] = -0.2
Mq[4,4] = -1.

Mq[1,2] = Mq[2,1] = 0.590
Mq[1,3] = Mq[3,1] = 0.766
Mq[2,4] = Mq[4,2] = 0.566
Mq[3,4] = Mq[4,3] = 0.460


@testset "testing marginal/conditional probabilities" begin
    β = 3.
    g = M2graph(Mq, -1)

    update_cells!(
      g,
      rule = square_lattice((1, 1, 2, 1, 2)),
    )

    fg = factor_graph(
        g,
        energy=energy,
        spectrum=brute_force,
    )

    states = collect.(all_states(rank_vec(g)))
    ρ = exp.(-β .* energy.(states, Ref(g)))
    Z = sum(ρ)

    println("grid of nodes")
    display([1; 2])
    println()

    peps = PepsNetwork(2,1, fg, β, :NW)
    Dcut = 8
    tol = 0.
    swep = 4
    z = contract(peps, Dict{Int, Int}())
    @test z ≈ Z

    boundary_mps = boundaryMPS(peps, Dcut, tol, swep)

    # initialize
    ps = Partial_sol{Float64}(Int[], 0.)

    obj1 = conditional_probabs(peps, ps, boundary_mps[1], PEPSRow(peps, 1))
    _, i = findmax(obj1)
    ps1 = update_partial_solution(ps, i, obj1[i])

    for i_1 in 1:4
        @test obj1[i_1] ≈ contract(peps, Dict{Int, Int}(1 => i_1))/z atol = 1e-3
    end

    obj2 = conditional_probabs(peps, ps1, boundary_mps[2], PEPSRow(peps, 2))
    obj2 = obj1[i].*obj2
    _, j = findmax(obj2)
    ps2 = update_partial_solution(ps1, j, obj2[j])

    for i_2 in 1:4
        @test obj2[i_2] ≈ contract(peps, Dict{Int, Int}(1 => i, 2 =>i_2))/z atol = 1e-3
    end
end
