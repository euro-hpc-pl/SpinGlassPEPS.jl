
function make_interactions_case1()
    J_h = [(1,1) 0.; (1,2) 0.; (1,4) 0.; (2,2) 0.; (2,3) 0.; (2,5) 0.; (3,3) -0.2; (3,6) 1.5]
    J_h = vcat(J_h, [(6,6) -2.2; (5,6) 0.; (6,9) -0.52; (5,5) 0.; (4,5) .0; (5,8) 0.0; (4,4) 0.; (4,7) 0.])
    J_h = vcat(J_h, [(7,7) 0.2; (7,8) 0.5; (8,8) -0.2; (8,9) -0.05; (9,9) -0.8])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

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

    grid = [1 2; 3 4]

    a =  Partial_sol{Float64}([1,1,1,2], 0.2)
    b = Partial_sol{Float64}([1,1,2,2], 1.)

    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    spins, objectives = return_solutions([a,b], ns)
    @test spins == [[-1, -1, 1, 1],[-1, -1, -1, 1]]
    @test objectives == [1.0, 0.2]
end

function make_interactions()
    J_h = [(1,1) 1.; (1,2) 2.; (1,4) -1.; (2,2) 1.4; (2,3) 1.1; (2,5) -.75; (3,3) -0.2; (3,6) 1.5]
    J_h = vcat(J_h, [(6,6) -2.2; (5,6) 2.1; (6,9) -0.52; (5,5) .2; (4,5) .5; (5,8) 0.12; (4,4) -1.; (4,7) -1.])
    J_h = vcat(J_h, [(7,7) 0.2; (7,8) 0.5; (8,8) -0.2; (8,9) -0.05; (9,9) -0.8])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "larger tensors" begin
    # TODO this will be done by the function
    grid = Array{Array{Int}}(undef, (2,2))
    grid[1,1] = [1 2;4 5]
    grid[1,2] = reshape([3, 6], (2,1))
    grid[2,1] = reshape([7, 8], (1,2))
    grid[2,2] = reshape([9], (1,1))
    grid = Array{Array{Int}}(grid)
    M = [1 2;3 4]

    q = make_interactions()
    β = 1.5
    ns = [Node_of_grid(i, M, grid) for i in 1:maximum(M)]
    T1 = compute_single_tensor(ns, q, 1, β)
    T2 = compute_single_tensor(ns, q, 2,  β)

    M = [1 2 3; 4 5 6; 7 8 9]
    ns = [Node_of_grid(i, M) for i in 1:maximum(M)]
    t1 = compute_single_tensor(ns, q, 1, β)
    t2 = compute_single_tensor(ns, q, 2, β)
    t3 = compute_single_tensor(ns, q, 3, β)
    t4 = compute_single_tensor(ns, q, 4, β)
    t5 = compute_single_tensor(ns, q, 5, β)
    t6 = compute_single_tensor(ns, q, 6, β)

    tensors = [t1[1,:,:,:], t2, t4[1,:,:,:,:], t5]
    modes = [[1,2,-10], [1, -1, 3, -11], [4, 2, -3, -12], [4, -2, 3, -4, -13]]

    tcompare = ncon(tensors, modes)
    @test T1[1,:,:,:] ≈ reshape(tcompare, (4,4,16))

    tensors = [t3[:,1,:,:], t6[:,1,:,:,:]]
    modes = [[-1,1,-10], [-2, 1, -3, -11]]

    tcompare = ncon(tensors, modes)
    @test T2[:,1,:,:] ≈ reshape(tcompare, (4,2,4))
end

# chimera cells, tha small one 2x2
function make_c2()
    J_h = [(1,1) 1.; (2,2) 1.4; (3,3) 0.5; (4,4) -1.; (1,2) 1.1; (1,4) -0.5; (2,3) 3.1; (3,4) -2.]
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

#normal chimera cell 8x2
function make_c8()
    h = [(1,1) 1.; (2,2) 1.; (3,3) 1.; (4,4) 1.; (5,5) 1.; (6,6) 1.; (7,7) 1.; (8,8) 1.]
    J = vcat(h, [(1,2) 0.1; (1,4) 0.1; (1,6) 0.1; (1,8) 0.1])
    J = vcat(J, [(3,2) 0.1; (3,4) 0.1; (3,6) 0.1; (3,8) 0.1])
    J = vcat(J, [(5,2) 0.1; (5,4) 0.1; (5,6) 0.1; (5,8) 0.1])
    J = vcat(J, [(7,2) 0.1; (7,4) 0.1; (7,6) 0.1; (7,8) 0.1])
    [Interaction(J[i,1], J[i,2]) for i in 1:size(J, 1)]
end

@testset "chimera cel" begin
    # TODO this will be done by the function as well
    grid = Array{Array{Int}}(undef, (1,1))
    grid[1,1] = [1 2;3 4]
    grid = Array{Array{Int}}(grid)
    M = reshape([1], (1,1))
    ns = [Node_of_grid(1,M, grid; chimera = true)]
    q = make_c2()

    grid1 = Array{Array{Int}}(undef, (1,1))
    grid1[1,1] = [1 2;4 3]
    grid1 = Array{Array{Int}}(grid1)
    ns_1 = [Node_of_grid(1,M, grid1)]

    β = 1.
    ten = compute_single_tensor(ns, q, 1, β)

    ten1 = compute_single_tensor(ns_1, q, 1, β)

    # spins are reshufled
    @test ten[1,1,1,1:4] ≈ ten1[1,1,1,1:4]
    @test ten[1,1,1,13:16] ≈ ten1[1,1,1,13:16]
    @test ten[1,1,1,5:8] ≈ ten1[1,1,1,9:12]
    @test ten[1,1,1,9:12] ≈ ten1[1,1,1,5:8]

    grid2 = Array{Array{Int}}(undef, (1,1))
    grid2[1,1] = [1 2;3 4;5 6;7 8]
    grid2 = Array{Array{Int}}(grid2)
    M = reshape([1], (1,1))
    ns8 = [Node_of_grid(1,M, grid2; chimera = true)]
    q8 = make_c8()

    β = 1.
    ten8 = compute_single_tensor(ns8, q8, 1, β)

    @test size(ten8) == (1, 1, 1, 256)
    @test ten8[1,1,1,1] ≈ exp(β*(-8+16*2*0.1))
    @test ten8[1,1,1,256] ≈ exp(β*(8+16*2*0.1))
    @test ten8[1,1,1,2] ≈ exp(β*(-6+8*2*0.1))
end

# TODO we may not want to have the large dims tensor output,
# some set_spins may be heelpfull
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

@testset "PEPS network vs encon" begin
    ints = make_interactions()
    grid = [1 2 3 ; 4 5 6 ; 7 8 9]
    @test v2energy(ones(2,2), [1,1]) == 4

    Mat = interactions2M(ints)
    ns = [Node_of_grid(i, grid) for i in 1:9]
    β = 3.
    M = Array{Union{Nothing, Array{Float64}}}(nothing, (3,3))
    k = 0
    for i in 1:3
        for j in 1:3
            k = k+1
            M[i,j] = compute_single_tensor(ns, ints, k, β)
        end
    end
    M = Matrix{Array{Float64, N} where N}(M)
    cc = contract3x3by_ncon(M)
    # testing peps creation

    v = [-1 for _ in 1:9]

    @test exp.(β*v2energy(Mat, v)) ≈ cc[1,1,1,1,1,1,1,1,1]

    v[1] = 1
    @test exp.(β*v2energy(Mat, v)) ≈ cc[2,1,1,1,1,1,1,1,1]

    v = [1, -1, 1, -1, 1, -1, 1, -1, 1]
    @test exp.(β*v2energy(Mat, v)) ≈ cc[2,1,2,1,2,1,2,1,2]
end

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

    interactions = make_interactions()
    grid = [1 2 3; 4 5 6; 7 8 9]
    β = 3.

    ns = [Node_of_grid(i, grid) for i in 1:9]
    M = Array{Union{Nothing, Array{Float64}}}(nothing, (3,3))
    k = 0
    for i in 1:3
        for j in 1:3
            k = k+1
            M[i,j] = compute_single_tensor(ns, interactions, k, β)
        end
    end
    M = Matrix{Array{Float64, N} where N}(M)

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
    lower_mps = make_lower_mps(grid, ns, interactions, 2, β, 0, 0.)
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

    lower_mps = make_lower_mps(grid, ns, interactions, 3, β ,0, 0.)
    M_temp = [M[2,i][:,:,sol2[i],:,:] for i in 1:3]
    obj2 = conditional_probabs(M_temp, lower_mps, sol2[end], sol2[4:4])
    # this is exact
    @test [cond1, cond2] ≈ obj2

    # with approximation marginal
    lower_mps_a = make_lower_mps(grid, ns, interactions, 2, β, 2, 1e-6)
    objective = conditional_probabs(A, lower_mps_a, 0, sol)
    @test objective ≈ [p1, p2]

    objective1 = conditional_probabs(A, lower_mps_a, sol1[end], sol1)
    @test objective1 ≈ [p11/p1, p12/p1]

    lower_mps_a = make_lower_mps(grid, ns, interactions, 3, β, 2, 1.e-6)
    obj2_a = conditional_probabs(M_temp, lower_mps_a, sol2[end], sol2[4:4])
    # this is approx
    @test [cond1, cond2] ≈ obj2_a

end


function interactions_case2(T::Type = Float64)
    css = 2.
    J_h = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
    J_h = vcat(J_h, [(6,6) 0.; (6,5) -1.75; (6,9) 0.; (5,5) 0.75; (5,4) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    J_h = vcat(J_h, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
    [Interaction{T}(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

function make_interactions_case3()
    css = 2.
    J_h = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
    J_h = vcat(J_h, [(6,6) 0.; (5,6) -1.75; (6,9) 0.; (5,5) 0.75; (4,5) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    J_h = vcat(J_h, [(7,7) 0.1; (7,8) 0.; (8,8) 0.1; (8,9) 0.; (9,9) 0.1])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "PEPS - solving simple train problem" begin

    inter = interactions_case2()
    # TODO forming a frid inside a solver
    grid = [1 2 3; 4 5 6; 7 8 9]
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    spins, objective = solve(inter, ns, grid, 4; β = 1., χ = 2)

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

    # here we give a little Jii to 7,8,9 q-bits to allow there for 8 additional
    # combinations and degeneracy
    permuted_ints = make_interactions_case3()

    grid = [1 2 3; 4 5 6; 7 8 9]
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    M = [1 2;3 4]
    grid1 = Array{Array{Int}}(undef, (2,2))
    grid1[1,1] = [1 2; 4 5]
    grid1[1,2] = reshape([3; 6], (2,1))
    grid1[2,1] = reshape([7; 8], (1,2))
    grid1[2,2] = reshape([9], (1,1))
    grid1 = Array{Array{Int}}(grid1)

    ns_large = [Node_of_grid(i, M, grid1) for i in 1:maximum(M)]

    spins, objective = solve(permuted_ints, ns, grid, 16; β = 1., threshold = 0.)
    spins_l, objective_l = solve(permuted_ints, ns_large, M, 16; β = 1., threshold = 0.)


    @test spins_l[1] == spins[1]
    @test spins[1] == [1, -1, 1, 1, -1, 1, 1, 1, 1]
    @test objective[1] ≈ 0.12151449832031348

    first_deg = [[1, -1, 1, 1, -1, 1, -1, 1, 1], [1, -1, 1, 1, -1, 1, 1, -1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1]]
    @test spins[2] in first_deg
    @test spins[3] in first_deg
    @test spins[4] in first_deg

    @test spins_l[2] in first_deg
    @test spins_l[3] in first_deg
    @test spins_l[4] in first_deg

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

    @test objective[5] ≈ 0.08145360410807015
    @test objective[6] ≈ 0.08145360410807015
    @test objective[7] ≈ 0.08145360410807015

    @test spins[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test spins_l[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test objective[8] ≈ 0.06668857063231606

    for i in 1:10
        @test objective[i] ≈ objective_l[i]
    end


    @testset "itterative approximatimation in solution" begin

        ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]
        spins_a, objective_a = solve(permuted_ints, ns, grid, 16; β = 1., χ = 2)

        spins_a[1] == [1, -1, 1, 1, -1, 1, 1, 1, 1]
        objective_a[1] ≈ 0.12151449832031348

        first_deg = [[1, -1, 1, 1, -1, 1, -1, 1, 1], [1, -1, 1, 1, -1, 1, 1, -1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1]]
        @test spins_a[2] in first_deg
        @test spins_a[3] in first_deg
        @test spins_a[4] in first_deg
        @test objective_a[2] ≈ 0.09948765671968342
        @test objective_a[3] ≈ 0.09948765671968342
        @test objective_a[4] ≈ 0.09948765671968342

        second_deg = [[1, -1, 1, 1, -1, 1, -1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, 1], [1, -1, 1, 1, -1, 1, 1, -1, -1]]
        @test spins_a[5] in second_deg
        @test spins_a[6] in second_deg
        @test spins_a[7] in second_deg
        @test objective_a[5] ≈ 0.08145360410807015
        @test objective_a[6] ≈ 0.08145360410807015
        @test objective_a[7] ≈ 0.08145360410807015

        @test spins_a[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
        @test objective_a[8] ≈ 0.06668857063231606
    end

    @testset "PEPS  - solving it on Float32" begin
        T = Float32

        ints = interactions_case2(T)

        grid = [1 2 3; 4 5 6; 7 8 9]

        ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]
        spins, objective = solve(ints, ns, grid, 4; β = T(2.), χ = 2, threshold = T(1e-6))

        #ground
        @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

        #first
        @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]

        @test typeof(objective[1]) == Float32
    end
end
function make_interactions_large()
    J_h = [(1,1) 2.8; (1,2) -0.3; (1,5) -0.2; (2,2) -2.7; (2,3) -0.255; (2,6) -0.21; (3,3) 2.6; (3,4) -0.222; (3,7) -0.213; (4,4) -2.5; (4,8) -0.2]
    J_h = vcat(J_h, [(5,5) 2.4; (5,6) -0.15; (5,9) -0.211; (6,6) -2.3; (6,7) -0.2; (6,10) -0.15; (7,7) 2.2; (7,8) -0.11; (7,11) -0.35; (8,8) -2.1; (8,12) -0.19])
    J_h = vcat(J_h, [(9,9) 2.; (9,10) -0.222; (9,13) -0.15; (10,10) -1.9; (10,11) -0.28; (10,14) -0.21; (11,11) 1.8; (11,12) -0.19; (11,15) -0.18; (12,12) -1.7; (12,16) -0.27])
    J_h = vcat(J_h, [(13,13) 1.6; (13,14) -0.32; (14,14) -1.5; (14,15) -0.19; (15,15) 1.4; (15,16) -0.21; (16,16) -1.3])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "larger system " begin

    int_larger = make_interactions_large()

    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]
    spins, objective = solve(int_larger, ns, grid, 10; β = 3., χ = 2, threshold = 1e-11)
    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    M = [1 2;3 4]
    grid1 = Array{Array{Int}}(undef, (2,2))
    grid1[1,1] = [1 2; 5 6]
    grid1[1,2] = [3 4; 7 8]
    grid1[2,1] = [9 10; 13 14]
    grid1[2,2] = [11 12; 15 16]
    grid1 = Array{Array{Int}}(grid1)

    ns_large = [Node_of_grid(i, M, grid1) for i in 1:maximum(M)]

    spins_l, objective_l = solve(int_larger, ns_large, M, 10; β = 3., χ = 2, threshold = 1e-11)
    for i in 1:10
        @test objective[i] ≈ objective_l[i]
        @test spins[i] == spins_l[i]
    end
end
function make_c2()
    h = [(1,1) .0; (2,2) 1.; (3,3) 1.; (4,4) 1.; (5,5) 1.; (6,6) 1.; (7,7) 1.; (8,8) 1.]
    J = vcat(h, [(1,2) 0.1; (1,4) 0.1; (2,3) 0.1; (3,4) 0.1])
    J = vcat(J, [(5,6) 0.1; (5,8) 0.1; (6,7) 0.1; (7,8) 0.1])
    J = vcat(J, [(1,5) 0.1; (3,7) 0.1])
    [Interaction(J[i,1], J[i,2]) for i in 1:size(J, 1)]
end

@testset "solving simple chimera cell" begin

    grid2 = Array{Array{Int}}(undef, (2,1))
    grid2[1,1] = [1 2;3 4]
    grid2[2,1] = [5 6;7 8]
    grid2 = Array{Array{Int}}(grid2)
    M = reshape([1;2], (2,1))
    ns = [Node_of_grid(i,M, grid2; chimera = true) for i in 1:2]
    q = make_c2()

    β = 1.
    ten = compute_single_tensor(ns, q, 1, β)

    spins, _ = solve(q, ns, M, 2; β=β)
    @test spins[1] == [1, 1, 1, 1, 1, 1, 1, 1]
    @test spins[2] == [-1, 1, 1, 1, 1, 1, 1, 1]

end
