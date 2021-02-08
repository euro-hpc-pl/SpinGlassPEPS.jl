import SpinGlassPEPS: Partial_sol, update_partial_solution, M2graph
import SpinGlassPEPS: energy, solve, contract, conditional_probabs, project_spin_from_above

Mq = zeros(4,4)
Mq[1,1] = 1.
Mq[2,2] = 1.4
Mq[3,3] = -0.2
Mq[4,4] = -1.

Mq[1,2] = Mq[2,1] = 0.590
Mq[1,3] = Mq[3,1] = 0.766
Mq[2,4] = Mq[4,2] = 0.566
Mq[3,4] = Mq[4,3] = 0.460

@testset "helpers" begin
    proj = [1. 0.; 0. 1.]
    mps_el = reshape([1.0*i for i in 1:8], (2,2,2))
    @test project_spin_from_above(proj, 2, mps_el) == [3. 7.; 4. 8.]

    mpo_el = reshape([1.0*i for i in 1:16], (2,2,2,2))
    @test project_spin_from_above(proj, 2, mpo_el)[:,:,1] == [3. 11.; 4. 12.]
    @test project_spin_from_above(proj, 2, mpo_el)[:,:,2] == [7. 15.; 8. 16.]
end

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

    # grid of nodes
    # [1; 2]

    peps = PepsNetwork(2, 1, fg, β, :NW)
    Dcut = 8
    tol = 0.
    swep = 4
    z = contract(peps, Dict{Int, Int}())
    @test z ≈ Z

    boundary_mps = boundaryMPS(peps, 2, Dcut, tol, swep)

    mb = boundary_mps[1]
    pr = PEPSRow(peps, 1)
    for i in 1:length(boundary_mps[1])
        @test size(mb[i], 2) == size(pr[i] ,4)
    end

    # initialize
    ps = Partial_sol{Float64}(Int[], 0.)

    peps_row = PEPSRow(peps, 1)
    mpo = MPO(peps, 1)

    obj1 = conditional_probabs(peps, ps, boundary_mps[1], mpo, peps_row)
    _, i = findmax(obj1)
    ps1 = update_partial_solution(ps, i, obj1[i])

    for i_1 in 1:4
        @test obj1[i_1] ≈ contract(peps, Dict{Int, Int}(1 => i_1))/z
    end

    peps_row = PEPSRow(peps, 2)
    mpo = MPO(peps, 2)

    obj2 = conditional_probabs(peps, ps1, boundary_mps[2], mpo, peps_row)
    obj2 = obj1[i].*obj2
    _, j = findmax(obj2)
    ps2 = update_partial_solution(ps1, j, obj2[j])

    for i_2 in 1:4
        @test obj2[i_2] ≈ contract(peps, Dict{Int, Int}(1 => i, 2 =>i_2))/z
    end
end
