@testset "PEPS - axiliary functions" begin

    # partial solution
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

    #grid = nxmgrid(2,2)

    a =  Partial_sol{Float64}([1,1,1,2], 0.2)
    b = Partial_sol{Float64}([1,1,2,2], 1.)

    M = [1. 1. 1. 0.; 1. 1. 0. 1.; 1. 0. 1. 1.; 0. 1. 1. 1.]

    g = M2graph(M)
    gg = graph4peps(g, (1,1))

    spins, objectives = return_solutions([a,b], gg)
    @test spins == [[-1, -1, 1, 1],[-1, -1, -1, 1]]
    @test objectives == [1.0, 0.2]
end

### creation a matrix of interactions step by step as an example
Mq = ones(4,4)
fullM2grid!(Mq, (2,2))

@testset "peps element" begin

    g = M2graph(Mq)
    β = 2.
    #smaller tensors
    g1 = graph4peps(g, (1,1))

    t1 = compute_single_tensor(g1, 1, β, sum_over_last = true)
    t2 = compute_single_tensor(g1, 2, β, sum_over_last = true)
    t3 = compute_single_tensor(g1, 3, β, sum_over_last = true)

    # all are on the egde from left or right

    @test size(t1) == (1, 2, 2)
    @test t1[1,:,:] ≈ [exp(-1*β) 0.; 0. exp(1*β)]

    @test size(t2) == (2,1,2)
    @test t2[:,1,:] ≈ [exp(1*β) exp(-1*β); exp(-3*β) exp(3*β)]

    @test size(t3) == (1,2,2)
    @test t3[1,:,:] ≈ [exp(1*β) exp(-3*β); exp(-1*β) exp(3*β)]

    t = compute_single_tensor(g1, 1, β)
    @test t1 ≈ t[:,:,:,1]+t[:,:,:,2]

    #used to construct a larger tensor of 2x2 nodes
    tensors = [t1, t2]
    modes = [[-1, 1,-3], [1, -2, -4]]
    T2 = ncon(tensors, modes)

    gg = graph4peps(g, (2,1))
    T1 = compute_single_tensor(gg, 1, β, sum_over_last = true)
    @test vec(T1) ≈ vec(T2)
end

# NCon constraction
function contract3x3by_ncon(M::Matrix{Array{T, N} where N}) where T <: AbstractFloat
    u1 = M[1,1][1,:,:,:]
    v1 = [2,31, -1]

    u2 = M[1,2][:,:,:,:]
    v2 = [2,3,32,-2]

    u3 = M[1,3][:,1,:,:]
    v3 = [3,33,-3]

    m1 = M[2,1][1,:,:,:,:]

    v4 = [4,  31, 41, -4]
    m2 = M[2,2]

    v5 = [4, 5, 32, 42, -5]
    m3 = M[2,3][:,1,:,:,:]

    v6 = [5, 33, 43, -6]

    d1 = M[3,1][1,:,:,:]

    v7 = [6, 41, -7]
    d2 = M[3,2][:,:,:,:]

    v8 = [6,7,42,-8]
    d3 = M[3,3][:,1,:,:]

    v9 = [7, 43, -9]

    tensors = (u1, u2, u3, m1, m2, m3, d1, d2, d3)
    indexes = (v1, v2, v3, v4, v5, v6, v7, v8, v9)

    ncon(tensors, indexes)
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

function form_peps(gg::MetaGraph, β::Float64, s::Tuple{Int, Int} = (3,3))
    M = Array{Union{Nothing, Array{Float64}}}(nothing, s)
    k = 0
    for i in 1:s[1]
        for j in 1:s[2]
            k = k+1
            M[i,j] = compute_single_tensor(gg, k, β)
        end
    end
    Matrix{Array{Float64, N} where N}(M)
end


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

# TODO this will be the ilostative step by step how does the probability computation work
@testset "testing marginal/conditional probabilities" begin

    ####   conditional probability implementation

    mps = MPSxMPO([ones(1,2,2), 2*ones(2,1,2)], [ones(1,2,1,2), ones(2,1,1,2)])
    @test mps == [2*ones(1,4,1), 4*ones(4,1,1)]

    a = scalar_prod_step(ones(2,2,2), ones(2,2,2), ones(2,2))
    @test a == [8.0 8.0; 8.0 8.0]

    a = scalar_prod_step(ones(2,2), ones(2,2,2), ones(2,2))
    @test a == [8.0, 8.0]

    v1 = [ones(1,2,2,2), ones(2,2,2,2), ones(2,2,2,2), ones(2,1,2,2)]
    v2 = [ones(1,2,2), ones(2,2,2), ones(2,2,2), ones(2,1,2)]
    a = conditional_probabs(v1, v2, 2, [2,2,2])
    @test a == [0.5, 0.5]

    a = conditional_probabs([ones(2,2), ones(2,2), ones(2,1)])
    @test a == [0.5, 0.5]

    β = 3.
    g = M2graph(Mq)
    gg = graph4peps(g, (1,1))

    M = form_peps(gg, β)

    #TODO make something with dimensionality
    cc = contract3x3by_ncon(M)
    su = sum(cc)

    # trace all spins
    m3 = [sum_over_last(el) for el in M[3,:]]
    m2 = [sum_over_last(el) for el in M[2,:]]
    m1 = [sum_over_last(el) for el in M[1,:]]
    # get it back to array{4}
    m1 = [reshape(el, (size(el,1), size(el,2), 1, size(el,3))) for el in m1]

    mx = MPSxMPO(m3, m2)
    mx = MPSxMPO(mx, m1)
    A = mx[1][:,:,1]
    B = mx[2][:,:,1]
    C = mx[3][:,:,1]
    @test (A*B*C)[1] ≈ su

    # probabilities
    A = Vector{Array{Float64, 4}}(M[1,:])

    row = 2
    lower_mps = make_lower_mps(gg, row, β, 0, 0.)
    # marginal prob
    sol = Int[]
    objective = conditional_probabs(A, lower_mps, 0, sol)
    p1 = sum(cc[1,:,:,:,:,:,:,:,:])/su
    p2 = sum(cc[2,:,:,:,:,:,:,:,:])/su
    # approx due to numerical accuracy
    @test objective ≈ [p1, p2]

    #conditional prob
    p11 = sum(cc[1,1,:,:,:,:,:,:,:])/su
    p12 = sum(cc[1,2,:,:,:,:,:,:,:])/su
    sol1 = Int[1]
    objective1 = conditional_probabs(A, lower_mps, sol1[end], sol1)
    # approx due to numerical accuracy
    @test objective1 ≈ [p11/p1, p12/p1]


    cond1 = sum(cc[1,2,1,2,1,:,:,:,:])/sum(cc[1,2,1,2,:,:,:,:,:])
    cond2 = sum(cc[1,2,1,2,2,:,:,:,:])/sum(cc[1,2,1,2,:,:,:,:,:])
    @test cond1+cond2 ≈ 1

    sol2 = Int[1,2,1,2]

    row = 3
    lower_mps = make_lower_mps(gg, row, β ,0, 0.)
    M_temp = [M[2,i][:,:,sol2[i],:,:] for i in 1:3]
    obj2 = conditional_probabs(M_temp, lower_mps, sol2[end], sol2[4:4])
    # this is exact
    @test [cond1, cond2] ≈ obj2

    # with approximation marginal
    row = 2
    lower_mps_a = make_lower_mps(gg, row, β, 2, 1e-6)
    objective = conditional_probabs(A, lower_mps_a, 0, sol)
    @test objective ≈ [p1, p2]

    objective1 = conditional_probabs(A, lower_mps_a, sol1[end], sol1)
    @test objective1 ≈ [p11/p1, p12/p1]

    row = 3
    lower_mps_a = make_lower_mps(gg, row, β, 2, 1.e-6)
    obj2_a = conditional_probabs(M_temp, lower_mps_a, sol2[end], sol2[4:4])
    # this is approx
    @test [cond1, cond2] ≈ obj2_a

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
