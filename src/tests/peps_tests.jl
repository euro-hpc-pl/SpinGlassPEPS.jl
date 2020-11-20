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

        a =  Partial_sol{Float64}([1,1,1,2], 0.2)
        b = Partial_sol{Float64}([1,1,2,2], 1.)

        M = [1. 1. 1. 0.; 1. 1. 0. 1.; 1. 0. 1. 1.; 0. 1. 1. 1.]

        g = M2graph(M)
        gg = graph4peps(g, (1,1))

        spins, objectives = return_solutions([a,b], gg)
        @test spins == [[-1, -1, 1, 1],[-1, -1, -1, 1]]
        @test objectives == [1.0, 0.2]

        # TODO test new functions here
    end
end

### creation a matrix of interactions step by step as an example
Mq = ones(4,4)
fullM2grid!(Mq, (2,2))

@testset "tensor construction" begin

    #TODO clear permutedims

    g = M2graph(Mq)
    β = 2.
    #smaller tensors
    g1 = graph4peps(g, (1,1))

    t1 = compute_single_tensor(g1, 1, β, sum_over_last = true)
    t2 = compute_single_tensor(g1, 2, β, sum_over_last = true)
    t3 = compute_single_tensor(g1, 3, β, sum_over_last = true)
    t1 = permutedims(t1, (1,3,2,4))
    t2 = permutedims(t2, (1,3,2,4))
    t3 = permutedims(t3, (1,3,2))
    # all are on the egde from left or right

    @test size(t1) == (1, 2, 1, 2)
    @test t1[1,:,1,:] ≈ [exp(-1*β) 0.; 0. exp(1*β)]

    @test size(t2) == (2,1,1,2)
    @test t2[:,1,1,:] ≈ [exp(1*β) exp(-1*β); exp(-3*β) exp(3*β)]

    @test size(t3) == (1,2,2)
    @test t3[1,:,:] ≈ [exp(1*β) exp(-3*β); exp(-1*β) exp(3*β)]

    t = compute_single_tensor(g1, 1, β)
    t = permutedims(t, (1,3,2,4,5))
    @test t1 ≈ t[:,:,:,:,1]+t[:,:,:,:,2]

    #used to construct a larger tensor of 2x2 nodes
    tensors = [t1[:,:,1,:], t2[:,:,1,:]]
    modes = [[-1, 1,-3], [1, -2, -4]]
    T2 = ncon(tensors, modes)

    gg = graph4peps(g, (2,1))
    T1 = compute_single_tensor(gg, 1, β, sum_over_last = true)
    T1 = permutedims(T1, (1,3,2))
    @test vec(T1) ≈ vec(T2)
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

    g = M2graph(Mq)
    gg = graph4peps(g, (1,1))


    ### forms a peps network
    β = 3.
    M = form_peps(gg, β)
    cc = contract3x3by_ncon(M)
    # testing peps creation

    v = [-1 for _ in 1:9]

    @test exp.(-β*v2energy(Mq, v)) ≈ cc[1,1,1,1,1,1,1,1,1]

    v[1] = 1
    @test exp.(-β*v2energy(Mq, v)) ≈ cc[2,1,1,1,1,1,1,1,1]

    v = [1, -1, 1, -1, 1, -1, 1, -1, 1]
    @test exp.(-β*v2energy(Mq, v)) ≈ cc[2,1,2,1,2,1,2,1,2]
end

# TODO this will be the ilustative step by step how does the probability computation work
@testset "testing marginal/conditional probabilities" begin

    ####   conditional probability implementation

    a = scalar_prod_step(ones(2,2,2), ones(2,2,2), ones(2,2))
    @test a == [8.0 8.0; 8.0 8.0]

    a = scalar_prod_step(ones(2,2), ones(2,2,2), ones(2,2))
    @test a == [8.0, 8.0]

    # inds from above have been set on v1 already
    v1 = [ones(1,2,2,2), ones(2,2,2,2), ones(2,2,2,2), ones(2,1,2,2)]
    v2 = MPS([ones(1,2,2), ones(2,2,2), ones(2,2,2), ones(2,2,1)])
    ind = [2,2,2]
    #               physical indices
    #
    #     |        |          |        |
    #  1--v1[1] -- v1[2]  -- v1[3] -- v1[4]--1
    #     |        |          |        |
    # 1--v2[1] -- v2[2]  --  v2[3]-- v2[4]--1
    #
    #                  ||
    #                  V
    #                                        |
    #   ind[1]   ind[2]    ind[3] ind[3]--v1[4]--1
    #     |       |            |             |
    # 1--v2[1] -- v2[2]  --  v2[3]     -- v2[4]--1


    A = set_spin_from_letf(v1[4:4], ind[3])


    a = conditional_probabs(A, v2, ind)
    @test a == [0.5, 0.5]

    #
    #    --  M -- M -- M -- 1
    #
    #
    M = [ones(2,2), ones(2,2), ones(2,1)]
    a = conditional_probabs(M)
    @test a == [0.5, 0.5]

    β = 3.
    g = M2graph(Mq)
    gg = graph4peps(g, (1,1))

    M = form_peps(gg, β)

    #TODO make something with dimensionality
    cc = contract3x3by_ncon(M)
    su = sum(cc)


    # first row case
    A = Vector{Array{Float64, 4}}([e[:,:,1,:,:] for e in M[1,:]])

    # the row for lower_mps
    row = 2
    lower_mps = make_lower_mps(gg, row, β, 0, 0.)

    # marginal prob
    sol = Int[]
    Al = set_spin_from_letf(A, 1)
    objective = conditional_probabs(Al, lower_mps, sol)

    p1 = sum(cc[1,:,:,:,:,:,:,:,:])/su
    p2 = sum(cc[2,:,:,:,:,:,:,:,:])/su
    # approx due to numerical accuracy
    @test objective ≈ [p1, p2]

    #conditional prob
    p11 = sum(cc[1,1,:,:,:,:,:,:,:])/su
    p12 = sum(cc[1,2,:,:,:,:,:,:,:])/su
    sol1 = Int[1]
    Al = set_spin_from_letf(A[2:end], 1)
    objective1 = conditional_probabs(Al, lower_mps, sol1)
    # approx due to numerical accuracy
    @test objective1 ≈ [p11/p1, p12/p1]

end

function interactions_case2()
    L = 9
    css = -2.
    J_h = [1 1 -1.25; 1 2 1.75; 1 4 css; 2 2 -1.75; 2 3 1.75; 2 5 0.; 3 3 -1.75; 3 6 css]
    J_h = vcat(J_h, [6 6 0.; 6 5 1.75; 6 9 0.; 5 5 -0.75; 5 4 1.75; 5 8 0.; 4 4 0.; 4 7 0.])
    J_h = vcat(J_h, [7 7 css; 7 8 0.; 8 8 css; 8 9 0.; 9 9 css])

    ig = MetaGraph(L, 0.0)

    set_prop!(ig, :description, "The Ising model.")

    for k in 1:size(J_h, 1)
        i, j, v = J_h[k,:]
        v = -v
        i = Int(i)
        j = Int(j)
        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
        end
    end
    ig
end


@testset "PEPS - solving simple train problem" begin

    g = interactions_case2()

    spins, objective = solve(g, 4; β = 1., χ = 2)

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]


    # introduce degeneracy
    set_prop!(g, 7, :h, 0.1)
    set_prop!(g, 8, :h, 0.1)
    set_prop!(g, 9, :h, 0.1)

    spins, objective = solve(g, 16; β = 1., threshold = 0.)
    spins_l, objective_l = solve(g, 16; β = 1., threshold = 0., node_size = (2,2))
    spins_a, objective_a = solve(g, 16; β = 1., χ = 2)


    @test spins_l[1] == spins[1]
    @test spins_a[1] == spins[1]
    @test spins[1] == [1, -1, 1, 1, -1, 1, 1, 1, 1]
    @test objective[1] ≈ 0.12151449832031348
    @test objective_a[1] ≈ 0.12151449832031348

    first_deg = [[1, -1, 1, 1, -1, 1, -1, 1, 1], [1, -1, 1, 1, -1, 1, 1, -1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1]]
    @test spins[2] in first_deg
    @test spins[3] in first_deg
    @test spins[4] in first_deg

    @test spins_l[2] in first_deg
    @test spins_l[3] in first_deg
    @test spins_l[4] in first_deg

    @test spins_a[2] in first_deg
    @test spins_a[3] in first_deg
    @test spins_a[4] in first_deg

    @test objective[2] ≈ 0.09948765671968342
    @test objective[3] ≈ 0.09948765671968342
    @test objective[4] ≈ 0.09948765671968342

    second_deg = [[1, -1, 1, 1, -1, 1, -1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, 1], [1, -1, 1, 1, -1, 1, 1, -1, -1]]
    @test spins[5] in second_deg
    @test spins[6] in second_deg
    @test spins[7] in second_deg

    @test spins_l[5] in second_deg
    @test spins_l[6] in second_deg
    @test spins_l[7] in second_deg

    @test spins_a[5] in second_deg
    @test spins_a[6] in second_deg
    @test spins_a[7] in second_deg

    @test objective[5] ≈ 0.08145360410807015
    @test objective[6] ≈ 0.08145360410807015
    @test objective[7] ≈ 0.08145360410807015

    @test spins[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test spins_l[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test spins_a[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test objective[8] ≈ 0.06668857063231606

    for i in 1:10
        @test objective[i] ≈ objective_l[i]
        @test objective[i] ≈ objective_a[i]
    end
end


@testset "PEPS  - solving it on Float32" begin
    T = Float32
    g = interactions_case2()

    spins, objective = solve(g, 4; β = T(2.), χ = 2, threshold = 1e-6)

    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]

    @test typeof(objective[1]) == Float32
end

function make_interactions_large()
    L = 16
    J_h = [1 1 -2.8; 1 2 0.3; 1 5 0.2; 2 2 2.7; 2 3 0.255; 2 6 0.21; 3 3 -2.6; 3 4 0.222; 3 7 0.213; 4 4 2.5; 4 8 0.2]
    J_h = vcat(J_h, [5 5 -2.4; 5 6 0.15; 5 9 0.211; 6 6 2.3; 6 7 0.2; 6 10 0.15; 7 7 -2.2; 7 8 0.11; 7 11 0.35; 8 8 2.1; 8 12 0.19])
    J_h = vcat(J_h, [9 9 -2.; 9 10 0.222; 9 13 0.15; 10 10 1.9; 10 11 0.28; 10 14 0.21; 11 11 -1.8; 11 12 0.19; 11 15 0.18; 12 12 1.7; 12 16 0.27])
    J_h = vcat(J_h, [13 13 -1.6; 13 14 0.32; 14 14 1.5; 14 15 0.19; 15 15 -1.4; 15 16 0.21; 16 16 1.3])

    ig = MetaGraph(L, 0.0)

    set_prop!(ig, :description, "The Ising model.")

    for k in 1:size(J_h, 1)
        i, j, v = J_h[k,:]
        v = -v
        i = Int(i)
        j = Int(j)
        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
        end
    end
    ig
end

@testset "larger system " begin

    g = make_interactions_large()
    spins, objective = solve(g, 10; β = 3., χ = 2, threshold = 1e-11)
    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    spins_l, objective_l = solve(g, 10; β = 3., χ = 2, threshold = 1e-11, node_size = (2,2))
    for i in 1:10
        @test objective[i] ≈ objective_l[i] atol=1e-8
        @test spins[i] == spins_l[i]
    end
end
