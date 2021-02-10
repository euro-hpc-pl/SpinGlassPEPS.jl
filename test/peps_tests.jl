include("test_helpers.jl")
import SpinGlassPEPS: Partial_sol, update_partial_solution, select_best_solutions
import SpinGlassPEPS: conditional_probabs
import SpinGlassPEPS: energy, solve
import SpinGlassPEPS: dX_inds, merge_dX

Random.seed!(1234)


@testset "factor graph and peps formation" begin

    β = 3.
    g = make_interactions_case2()

    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )
    @test props(fg, 1)[:cluster].vertices == Dict(1 => 1)
    @test nv(fg) == 16
    @test ne(fg) == 24

    peps = PepsNetwork(4, 4, fg, β, :NW)
    @test peps.size == (4,4)
    @test peps.i_max == 4
    @test peps.j_max == 4

    #  T1 -- T2
    #  |     |
    #  T3 -- T4
    #            .
    #   1 -- 2 --.-- 3 -- 4
    #   |    |   .   |    |
    #   5 -- 6 --.-- 7 -- 8
    #   |    |   .   |    |
    # .......................
    #   |    |   .   |    |
    #   9 -- 10 -.-- 11 --12
    #   |     |  .    |    |
    #   13 --14 -.-- 15 --16
    #

    ns = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    g = make_interactions_case2()

    update_cells!(
      g,
      rule = square_lattice((2, 2, 2, 2, 1)),
    )

    fg = factor_graph(
        g,
        16,
        energy=energy,
        spectrum=brute_force,
    )


    D = props(fg, 1)[:cluster].vertices
    @test sort([v for v in values(D)]) == [1,2,3,4]
    nodes = [e for e in keys(D)]
    @test sort(nodes) == sort(vec(ns[1:2, 1:2]))

    D = props(fg, 2)[:cluster].vertices
    nodes = [e for e in keys(D)]
    @test sort(nodes) == sort(vec(ns[1:2, 3:4]))

    D = props(fg, 3)[:cluster].vertices
    nodes = [e for e in keys(D)]
    @test sort(nodes) == sort(vec(ns[3:4, 1:2]))

    D = props(fg, 4)[:cluster].vertices
    nodes = [e for e in keys(D)]
    @test sort(nodes) == sort(vec(ns[3:4, 3:4]))

    @test nv(fg) == 4
    @test ne(fg) == 4

    peps = PepsNetwork(2, 2, fg, β, :NW)
    @test peps.size == (2,2)
end


@testset "factor graph 3 x 3" begin
    #this is full graph
    M = ones(9,9)
    #this is grid of size 3x3
    fullM2grid!(M, (3,3))

    # change it to Ising
    g = M2graph(M)

    β = 3.

    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )

    @test nv(fg) == 9
    @test ne(fg) == 12

    peps = PepsNetwork(3,3, fg, β, :NW)
    @test peps.size == (3,3)

    rule = Dict{Any,Any}(1 => 1, 2 => 1, 4 => 1, 5 => 1, 3=>2, 6 => 2, 7 => 3, 8 => 3, 9 => 4)

    update_cells!(
      g,
      rule = rule,
    )

    fg = factor_graph(
        g,
        16,
        energy=energy,
        spectrum=brute_force,
    )

    @test nv(fg) == 4
    @test ne(fg) == 4

    peps = PepsNetwork(2,2, fg, β, :NW)
    @test peps.size == (2,2)
end

@testset "partial solution type" begin
    ps = Partial_sol{Float64}()
    @test ps.spins == []
    @test ps.objective == 1.

    ps1 = Partial_sol{Float64}([1,1], 1.)
    @test ps1.spins == [1,1]
    @test ps1.objective == 1.

    ps2 = update_partial_solution(ps1, 2, 1.)
    @test ps2.spins == [1,1,2]
    @test ps2.objective == 1.

    ps3 = Partial_sol{Float64}([1,1,1], .2)

    b = select_best_solutions([ps3, ps2], 1)
    @test b[1].spins == [1, 1, 2]
    @test b[1].objective == 1.
end


@testset "droplet hepers" begin

    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12]
    i = dX_inds(size(grid, 2), 2)
    @test i == [1]
    i = dX_inds(size(grid, 2), 1)
    @test i == Int[]

    # 1   2     3    4
    #        ?  |    |
    # 5   6    <7>   8
    # |   |
    # 9   10   11   12
    #

    i = dX_inds(size(grid, 2), 7)
    @test i == [3, 4, 5, 6]

    i = dX_inds(size(grid, 2), 7; has_diagonals = true)
    @test i == [2, 3, 4, 5, 6]


    # 5     6     7   8
    # |     |     |    |
    # <9>   10   11   12
    #
    #both cases the same
    i = dX_inds(size(grid, 2), 9)
    @test i == [5,6,7,8]

    i = dX_inds(size(grid, 2), 9; has_diagonals = true)
    @test i == [5,6,7,8]

    # other grid

    grid1 = [1 2; 3 4; 5 6; 7 8]
    i = dX_inds(size(grid1, 2), 5)
    @test i == [3,4]

    a = Partial_sol{Float64}([1,1,1], 0.2)
    b = Partial_sol{Float64}([2,1,1], 0.18)
    c = Partial_sol{Float64}([1,1,2], 1.)
    d = Partial_sol{Float64}([2,1,2], .1)

    vps = [a,b,c,d]

    boundary = [2,3]

    #ratio of objectives

    # 0.18/0.2 = 0.9
    # 0.1/1. = 0.1
    thershold = 0.15

    ps1 = merge_dX(vps, boundary, thershold)
    @test ps1 == [a,b,c]

    thershold = 0.95

    ps1 = merge_dX(vps, boundary, thershold)

    @test ps1 == [a,c]

    thershold = 0.

    ps1 = merge_dX(vps, boundary, thershold)
    @test ps1 == [a,b,c,d]
end


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


@testset "test an exemple instance" begin
    δH = 1e-6
    β = 3.
    g = make_interactions_case2()

    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(4, 4, fg, β, :NW)

    sols = solve(peps, 10; β = β)

    @test sols[1].spins == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    objective = [e.objective for e in sols]

    spins = return_solution(g, fg, sols)

    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]


    update_cells!(
      g,
      rule = square_lattice((2, 2, 2, 2, 1)),
    )

    fg = factor_graph(
        g,
        16,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(2, 2, fg, β, :NW)

    sols = solve(peps, 10; β = β, χ = 2, threshold = 1e-11, δH = δH)
    objective_l = [e.objective for e in sols]
    spins_l = return_solution(g, fg, sols)
    for i in 1:10

        @test objective[i] ≈ objective_l[i] atol=1e-8
        @test spins[i] == spins_l[i]
    end
    # low energy spectrum

    g1 = make_interactions_case2()

    update_cells!(
      g1,
      rule = square_lattice((2, 2, 2, 2, 1)),
    )
    
    D = Dict{Int, Int}()
    for v in vertices(g1)
        push!(D, (v => 15))
    end
    fg = factor_graph(
        g1,
        D,
        energy=energy,
        spectrum=brute_force,
    )
    peps = PepsNetwork(2,2, fg, β, :NW)

    sols = solve(peps, 10; β = β, χ = 2, threshold = 1e-11, δH = δH)
    objective_s = [e.objective for e in sols]
    spins_s = return_solution(g1, fg, sols)

    for i in 1:10
        @test objective[i] ≈ objective_s[i] atol=1e-8
        @test spins[i] == spins_s[i]
    end
end

# TODO chech different types of Float
#=
@testset "test an exemple instance on Float32" begin
    δH = 1e-6
    T = Float32
    g = make_interactions_case2(T)


    β = T(3.)
    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(4, 4, fg, β, :NW)

    sols = solve(g, peps, 10; β = β)
    objective = [e.objective for e in sols]
    spins = return_solution(g, fg, sols)

    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    update_cells!(
      g,
      rule = square_lattice((2, 2, 2, 2, 1)),
    )

    fg = factor_graph(
        g,
        16,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(2, 2, fg, β, :NW)

    sols = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11)

    objective_l = [e.objective for e in sols]
    spins_l = return_solution(g, fg, sols)
    for i in 1:10
        @test objective[i] ≈ objective_l[i] atol=1e-5
        @test spins[i] == spins_l[i]
    end
end
=#
