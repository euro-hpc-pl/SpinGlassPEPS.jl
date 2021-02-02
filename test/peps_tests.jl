include("test_helpers.jl")
import SpinGlassPEPS: Partial_sol, update_partial_solution, select_best_solutions, return_solutions
import SpinGlassPEPS: compute_single_tensor, conditional_probabs, get_parameters_for_T
import SpinGlassPEPS: make_lower_mps, M2graph, graph4peps, fullM2grid!
import SpinGlassPEPS: set_spin_from_letf, spin_index_from_left, spin_indices_from_above
import SpinGlassPEPS: energy, solve
import SpinGlassPEPS: dX_inds, merge_dX
import SpinGlassPEPS: reshape_row
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
    println(sort([v for v in values(D)]) == [1,2,3,4])
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
    display(M)
    println()
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

    @test nv(fg) == 4
    @test ne(fg) == 4

    peps = PepsNetwork(2,2, fg, β, :NW)
    @test peps.size == (2,2)
end



@testset "PEPS - axiliary functions" begin

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

    @testset "functions of graph" begin

        a = Partial_sol{Float64}([1,1,1,2], 0.2)
        b = Partial_sol{Float64}([1,1,2,2], 1.)

        M = [1. 1. 1. 0.; 1. 1. 0. 1.; 1. 0. 1. 1.; 0. 1. 1. 1.]

        g = M2graph(M)
        gg = graph4peps(g, (1,1))

        spins, objectives = return_solutions([a,b], gg)
        @test spins == [[-1, -1, 1, 1], [-1, -1, -1, 1]]
        @test objectives == [1.0, 0.2]

        M = ones(16,16)
        fullM2grid!(M, (4,4))
        g = M2graph(M)
        gg = graph4peps(g, (2,2))

        ps = Partial_sol{Float64}([6], .2)
        ul,ur = spin_indices_from_above(gg, ps, 2)
        l = spin_index_from_left(gg, ps, 2)
        @test ul == [2]
        @test ur == [1]
        @test l == 2

        ps = Partial_sol{Float64}([4,6], .2)
        ul,ur = spin_indices_from_above(gg, ps, 3)
        l = spin_index_from_left(gg, ps, 3)
        @test ul == Int[]
        @test ur == [1,2]
        @test l == 1

    end

    @testset "droplet hepers" begin

        grid = [1 2 3 4; 5 6 7 8; 9 10 11 12]
        i = dX_inds(grid, 2)
        @test i == [1]
        i = dX_inds(grid, 1)
        @test i == Int[]

        # 1   2     3    4
        #        ?  |    |
        # 5   6    <7>   8
        # |   |
        # 9   10   11   12
        #

        i = dX_inds(grid, 7)
        @test i == [3, 4, 5, 6]

        i = dX_inds(grid, 7; has_diagonals = true)
        @test i == [2, 3, 4, 5, 6]


        # 5     6     7   8
        # |     |     |    |
        # <9>   10   11   12
        #
        #both cases the same
        i = dX_inds(grid, 9)
        @test i == [5,6,7,8]

        i = dX_inds(grid, 9; has_diagonals = true)
        @test i == [5,6,7,8]

        # other grid

        grid1 = [1 2; 3 4; 5 6; 7 8]
        i = dX_inds(grid1, 5)
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
end

### creation a matrix of interactions step by step as an example
Mq = ones(4,4)
fullM2grid!(Mq, (2,2))


@testset "tensor construction" begin

    g = M2graph(Mq)

    g_ising = M2graph(Mq)
    m = 2
    n = 2
    t = 1

    fg = factor_graph(
        g_ising,
        energy=energy,
        spectrum=full_spectrum,
    )

    β = 2.
    peps = PepsNetwork(2, 2,  fg, β, :NW)


    #smaller tensors
    g1 = graph4peps(g, (1,1))

    right, down, M_left, M_up = get_parameters_for_T(g1, 1)

    @test right == [1]
    @test down == [1]
    @test M_left == [0.0 0.0]
    @test M_up == [0.0 0.0]

    t11 = compute_single_tensor(g1, 1, β, sum_over_last = false)
    t1 = compute_single_tensor(g1, 1, β, sum_over_last = true)
    t2 = compute_single_tensor(g1, 2, β, sum_over_last = true)
    t3 = compute_single_tensor(g1, 3, β, sum_over_last = true)

    t12 = compute_single_tensor(g1, 2, β, sum_over_last = false)
    t13 = compute_single_tensor(g1, 3, β, sum_over_last = false)

    B = generate_tensor(peps, (1,1))
    @test B == t11

    update_cells!(
       g_ising,
       rule = square_lattice((1, 2, 1, 2, 1)),
    )

    fg = factor_graph(
        g_ising,
        energy=energy,
        spectrum=full_spectrum,
    )

    β = 2.

    peps = PepsNetwork(1, 1, fg, β, :NW)
    B = generate_tensor(peps, (1,1))

    gg = graph4peps(g, (2,2))
    T1 = compute_single_tensor(gg, 1, β, sum_over_last = true)

    @test sum(B) == T1[1]


    @test size(t1) == (1, 1, 2, 2)
    @test t1[1,1,:,:] ≈ [exp(1*β) 0.; 0. exp(-1*β)]

    @test size(t2) == (2,1,1,2)
    @test t2[:,1,1,:] ≈ [exp(-1*β) exp(1*β); exp(3*β) exp(-3*β)]

    @test size(t3) == (1,2,2,1)
    @test t3[1,:,:,1] ≈ [exp(-1*β) exp(1*β); exp(3*β) exp(-3*β)]

    t = compute_single_tensor(g1, 1, β)

    @test t1 ≈ t[:,:,:,:,1]+t[:,:,:,:,2]

    #used to construct a larger tensor of 2x2 nodes
    tensors = [t1[:,1,:,:], t2[:,1,:,:]]
    modes = [[-1, 1,-2], [1, -3, -4]]
    T2 = ncon(tensors, modes)

    gg = graph4peps(g, (2,1))
    T1 = compute_single_tensor(gg, 1, β, sum_over_last = true)

    p = [2,3,1,4]
    @test vec(T1) ≈ vec(T2)[p]
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


@testset "whole peps tensor" begin

    g = M2graph(Mq, -1)
    gg = graph4peps(g, (1,1))

    ps =[sortperm(props(gg, i)[:spectrum]) for i in 1:9]

    ### forms a peps network
    β = 2.
    M = form_peps(gg, β)
    cc = contract3x3by_ncon(M)
    # testing peps creation

    v = [-1 for _ in 1:9]
    ii = [p[1] for p in ps]

    @test exp.(-β*energy(v, g)) ≈ cc[ii...]

    v[1] = 1
    ii[1] = ps[1][2]
    @test exp.(-β*energy(v, g)) ≈ cc[ii...]

    v = [1 for _ in 1:9]
    ii = [p[2] for p in ps]
    @test exp.(-β*energy(v, g)) ≈ cc[ii...]

    fg = factor_graph(
        g,
        energy=energy,
        spectrum=full_spectrum,
    )

    peps = PepsNetwork(3, 3, fg, β, :NW)
    B = generate_tensor(peps, (1,1))

    mpo1 = MPO(PEPSRow(peps, 1))

    println(size(mpo1[1]))
    println(size(mpo1[2]))
    println(size(mpo1[3]))

    pp = PEPSRow(peps, 2)
    #println(pp)

    mpo2 = MPO(PEPSRow(peps, 2))

    mpo12 = mpo1*mpo2

    mpsu = MPS([permutedims(e[:,1,:,:], [1,3,2]) for e in mpo12])

    mpo3 = MPO(PEPSRow(peps, 3))

    mpsl = MPS([e[:,:,:,1] for e in mpo3])

    @test right_env(mpsu, mpsl)[end] ≈ [1.]
    g1 = copy(g)

    update_cells!(
       g1,
       rule = square_lattice((1, 3, 1, 3, 1)),
    )

    fg = factor_graph(
        g1,
        energy=energy,
        spectrum=full_spectrum,
    )

    peps = PepsNetwork(1, 1, fg, β, :NW)
    B = generate_tensor(peps, (1,1))
    println(size(B))
    println(size(vec(cc)))
    @test sum(cc) ≈ sum(B)
end

# TODO this will be the ilustative step by step how does the probability computation work

@testset "testing marginal/conditional probabilities" begin

    ####   conditional probability implementation


    mpo = MPO([ones(2,2,2,2), ones(2,2,2,2)])
    mps = set_spin_from_letf(mpo, 1)
    @test mps[1] == ones(2,2,2)
    @test mps[2] == 2*ones(2,2,2)

    β = 3.
    g = M2graph(Mq, -1)
    gg = graph4peps(g, (1,1))

    fg = factor_graph(
        g,
        energy=energy,
        spectrum=full_spectrum,
    )

    origin = :NW

    peps = PepsNetwork(3, 3, fg, β, origin)
    mpo2 = MPO(PEPSRow(peps, 2))
    mpo3 = MPO(PEPSRow(peps, 3))

    M = form_peps(gg, β)

    #TODO make something with dimensionality
    cc = contract3x3by_ncon(M)
    su = sum(cc)


    # first row
    A =  M[1,:]

    # the row for lower_mps
    row = 1
    lower_mps = make_lower_mps(gg, row+1, β, 0, 0.)

    l_mps = MPS([e[:,:,:,1] for e in mpo2*mpo3])

    #AA = MPO(peps, 1, false)

    println(size(A[1]))
    #println(size(AA[1]))

    # marginal prob
    sol = Partial_sol{Float64}(Int[], 0.)
    j = 1
    objective = conditional_probabs(gg, sol, j, lower_mps, A)
    println(objective)

    #objective1 = conditional_probabs(gg, sol, j, l_mps, AA)
    #println(objective1)

    sol = Partial_sol{Float64}([1], objective[1])

    p1 = sum(cc[1,:,:,:,:,:,:,:,:])/su
    p2 = sum(cc[2,:,:,:,:,:,:,:,:])/su
    # approx due to numerical accuracy
    @test objective ≈ [p1, p2]

    j = 2
    objective = conditional_probabs(gg, sol, j, lower_mps, A)
    #conditional prob
    p11 = sum(cc[1,1,:,:,:,:,:,:,:])/su
    p12 = sum(cc[1,2,:,:,:,:,:,:,:])/su
    # approx due to numerical accuracy
    @test objective ≈ [p11/p1, p12/p1]

    j = 5
    row = 2
    lower_mps = make_lower_mps(gg, row+1, β, 0, 0.)
    A =  M[2,:]

    # objective value from the previous step is set artificially
    sol = Partial_sol{Float64}(Int[1,1,1,1], 1.)
    objective = conditional_probabs(gg, sol, j, lower_mps, A)
    #conditional prob
    p1 = sum(cc[1,1,1,1,1,:,:,:,:])/sum(cc[1,1,1,1,:,:,:,:,:])
    p2 = sum(cc[1,1,1,1,2,:,:,:,:])/sum(cc[1,1,1,1,:,:,:,:,:])
    # approx due to numerical accuracy
    @test objective ≈ [p1, p2]
end


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

    spins, objective = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11, δH = δH)
    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    g1 = make_interactions_case2()

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

    spins_l, objective_l = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11, node_size = (2,2), δH = δH)
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

    fg = factor_graph(
        g1,
        15,
        energy=energy,
        spectrum=brute_force,
    )
    peps = PepsNetwork(2,2, fg, β, :NW)

    spins_s, objective_s = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11, node_size = (2,2), spectrum_cutoff = 15, δH = δH)
    for i in 1:10
        @test objective[i] ≈ objective_s[i] atol=1e-8
        @test spins[i] == spins_s[i]
    end
end


@testset "test an exemple instance on Float32" begin
    δH = 1e-6
    g = make_interactions_case2()

    T = Float32
    β = T(3.)
    fg = factor_graph(
        g,
        2,
        energy=energy,
        spectrum=brute_force,
    )

    peps = PepsNetwork(4, 4, fg, β, :NW)

    spins, objective = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11, δH = δH)
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

    spins_l, objective_l = solve(g, peps, 10; β = β, χ = 2, threshold = 1e-11, node_size = (2,2), δH = δH)
    for i in 1:10
        @test objective[i] ≈ objective_l[i] atol=1e-5
        @test spins[i] == spins_l[i]
    end
end
