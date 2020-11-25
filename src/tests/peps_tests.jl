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

        M = ones(16,16)
        fullM2grid!(M, (4,4))
        g = M2graph(M)
        gg = graph4peps(g, (2,2))

        ps = Partial_sol{Float64}([6], .2)
        ul,ur = spin_indices_from_above(gg, ps, 2)
        l = spin_index_from_left(gg, ps, 2)
        @test ul == [1]
        @test ur == [1]
        @test l == 2

        ps = Partial_sol{Float64}([4,6], .2)
        ul,ur = spin_indices_from_above(gg, ps, 3)
        l = spin_index_from_left(gg, ps, 3)
        @test ul == Int[]
        @test ur == [2,1]
        @test l == 1

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

    no_spins, tensor_size, right, down, M_left, M_up = get_parameters_for_T(g1, 1)
    @test no_spins == 1
    @test tensor_size == [1, 1, 2, 2, 2]
    @test right == [1]
    @test down == [1]
    @test M_left == [0.0 0.0]
    @test M_up == [0.0 0.0]

    t1 = compute_single_tensor(g1, 1, β, sum_over_last = true)
    t2 = compute_single_tensor(g1, 2, β, sum_over_last = true)
    t3 = compute_single_tensor(g1, 3, β, sum_over_last = true)

    t1 = permutedims(t1, (1,3,2,4))
    t2 = permutedims(t2, (1,3,2,4))
    t3 = permutedims(t3, (1,3,2,4))
    # all are on the egde from left or right

    @test size(t1) == (1, 2, 1, 2)
    @test t1[1,:,1,:] ≈ [exp(-1*β) 0.; 0. exp(1*β)]

    @test size(t2) == (2,1,1,2)
    @test t2[:,1,1,:] ≈ [exp(1*β) exp(-1*β); exp(-3*β) exp(3*β)]

    @test size(t3) == (1,2,2,1)
    @test t3[1,:,:,1] ≈ [exp(1*β) exp(-3*β); exp(-1*β) exp(3*β)]

    t = compute_single_tensor(g1, 1, β)
    t = permutedims(t, (1,3,2,4,5))
    @test t1 ≈ t[:,:,:,:,1]+t[:,:,:,:,2]

    #used to construct a larger tensor of 2x2 nodes
    tensors = [t1[:,:,1,:], t2[:,:,1,:]]
    modes = [[-1, 1,-3], [1, -2, -4]]
    T2 = ncon(tensors, modes)

    gg = graph4peps(g, (2,1))
    T1 = compute_single_tensor(gg, 1, β, sum_over_last = true)
    T1 = permutedims(T1, (1,3,2,4))
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

    mps1 = MPS([ones(2,2,2), ones(2,2,1)])
    mps2 = MPS([ones(2,2,2), ones(2,2,1)])
    @test compute_scalar_prod(mps1, mps2) == [16.0 16.0; 16.0 16.0]

    Ms = [ones(1,2), ones(2,2), ones(2,2)]
    @test Mprod(Ms) == [4.0 4.0]

    mpo = [ones(2,2,2,2), ones(2,2,2,2)]
    mps = set_spin_from_letf(mpo, 1)
    @test mps[1] == ones(2,2,2)
    @test mps[2] == 2*ones(2,2,2)

    β = 3.
    g = M2graph(Mq)
    gg = graph4peps(g, (1,1))

    M = form_peps(gg, β)

    #TODO make something with dimensionality
    cc = contract3x3by_ncon(M)
    su = sum(cc)


    # first row
    A =  M[1,:]

    # the row for lower_mps
    row = 1
    lower_mps = make_lower_mps(gg, row+1, β, 0, 0.)

    # marginal prob
    sol = Partial_sol{Float64}(Int[], 0.)
    j = 1
    objective = conditional_probabs(gg, sol, j, lower_mps, A)
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
