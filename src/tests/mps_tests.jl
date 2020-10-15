using TensorOperations


@testset "axiliary" begin
    b = scalar_prod_with_itself([ones(1,2,2), ones(2,1,2)])
    @test b == 16.0*ones(1,1)

    M = [1 2; 3 4]
    ns = [Node_of_grid(i, Array(transpose(M))) for i in 1:maximum(M)]

    i,j = connections_for_mps(ns)
    @test i == [1,2,3]
    @test j == [[3, 2], [4], [4]]
    a,b = cluster_conncetions(i,j)

    @test a == [[1], [2], [3]]
    @test b == [[[3, 2]], [[4]], [[4]]]

    grid = [1 2 3; 4 5 6; 7 8 9]
    A = Array(transpose(grid))

    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]
    i,j = connections_for_mps(ns)

    @test i == [1, 2, 3, 4, 5, 6, 7, 8]
    @test j == [[2, 4], [3, 5], [6], [5, 7], [6, 8], [9], [8], [9]]

    a,b = cluster_conncetions(i,j)
    @test a == [[1, 5], [2, 6], [3, 7], [4, 8]]
    @test b == [[[2, 4], [6, 8]], [[3, 5], [9]], [[6], [8]], [[5, 7], [9]]]
end

function make_qubo_x()
    qubo = [(1,1) .5; (1,2) -0.5; (1,4) -1.5; (2,2) -1.; (2,3) -1.5; (2,5) -0.5; (3,3) 2.; (3,6) 1.5]
    qubo = vcat(qubo, [(6,6) .05; (5,6) -0.25; (6,9) -0.52; (5,5) 0.75; (4,5) 0.5; (5,8) 0.5; (4,4) 0.; (4,7) -0.01])
    qubo = vcat(qubo, [(7,7) 0.35; (7,8) 0.5; (8,8) -0.08; (8,9) -0.05; (9,9) 0.33])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end


function make_qubo_circ()
    qubo = [(1,1) 0.; (1,2) 0.; (1,4) 2.; (2,2) 0.; (2,3) 2.; (2,5) 0.; (3,3) 0.; (3,6) 2.]
    qubo = vcat(qubo, [(6,6) 0.; (5,6) 0.; (6,9) 2.; (5,5) 0.; (4,5) 0.; (5,8) 0.; (4,4) 0.; (4,7) 2.])
    qubo = vcat(qubo, [(7,7) 0.; (7,8) 0.; (8,8) 0.; (8,9) .0; (9,9) 0.])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

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


@testset "MPS computing" begin

    qubo =  make_qubo_x()

    β = 2.

    mps = initialize_mps(9)
    @test mps[1] == ones(1,1,2)
    @test mps[2] == ones(1,1,2)
    @test mps[3] == ones(1,1,2)

    grid = [1 2 3; 4 5 6; 7 8 9]
    ns = [Node_of_grid(i,grid) for i in 1:maximum(grid)]

    is, js = connections_for_mps(ns)
    all_is, all_js = cluster_conncetions(is,js)

    mps = construct_mps(qubo, β, 2, 9, all_is, all_js, 4, 0.)

    grid = [1 2 3 ; 4 5 6; 7 8 9]

    ns = [Node_of_grid(i, grid) for i in 1:9]

    M = Array{Union{Nothing, Array{Float64}}}(nothing, (3,3))
    k = 0
    for i in 1:3
        for j in 1:3
            k = k+1
            M[i,j] = compute_single_tensor(ns, qubo, k, β)
        end
    end
    M = Matrix{Array{Float64, N} where N}(M)

    cc = contract3x3by_ncon(M)

    v = ones(1)*mps[1][:,:,1]*mps[2][:,:,1]
    v = reshape(v, size(v,2))
    v1 = v_from_mps(mps, [1,1])

    @test v == v1

    A = mps[3]
    B = zeros(2,2)
    M = scalar_prod_with_itself(mps[4:end])
    @tensor begin
        B[x,y] = A[a,b,x]*A[c,d,y]*v1[a]*v1[c]*M[b,d]
    end

    @test compute_probs(mps, [1,1]) ≈ diag(B)

    @test compute_probs(mps, [1,1]) ≈ sum(cc[1,1,:,:,:,:,:,:,:], dims = (2,3,4,5,6,7))
    @test compute_probs(mps, [1,1,2,2,1]) ≈ sum(cc[1,1,2,2,1,:,:,:,:], dims = (2,3,4))
    @test compute_probs(mps, [1,1,2,2,1,2,2,1]) ≈ cc[1,1,2,2,1,2,2,1,:]

    # approximation

    mps_a = construct_mps(qubo, β, 3, 9, all_is, all_js, 2, 1.e-12)
    ps = sum(cc[1,1,:,:,:,:,:,:,:], dims = (2,3,4,5,6,7))
    pp = compute_probs(mps, [1,1])

    @test ps/sum(ps) ≈ pp/sum(pp)

    ps = sum(cc[1,1,2,2,:,:,:,:,:], dims = (2,3,4,5))
    pp = compute_probs(mps, [1,1,2,2])

    @test ps/sum(ps) ≈ pp/sum(pp)

    ps = cc[1,1,2,2,1,2,1,2,:]
    pp = compute_probs(mps, [1,1,2,2,1,2,1,2])

    @test ps/sum(ps) ≈ pp/sum(pp)
end


function make_qubo(T::Type = Float64)
    css = 2.
    qubo = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
    qubo = vcat(qubo, [(6,6) 0.; (6,5) -1.75; (6,9) 0.; (5,5) 0.75; (5,4) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
    [Qubo_el{T}(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "MPS - solving simple train problem" begin

    train_qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]

    ns = [Node_of_grid(i, train_qubo) for i in 1:get_system_size(train_qubo)]

    spins, _ = solve_mps(train_qubo, ns, 2; β=2., β_step=2, χ=2, threshold = 1e-14)

    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]

    function make_qubo1()
        css = 2.
        qubo = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) -1.75; (6,9) 0.; (5,5) 0.75; (4,5) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.1; (7,8) 0.; (8,8) 0.1; (8,9) 0.; (9,9) 0.1])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    permuted_train_qubo = make_qubo1()

    ns = [Node_of_grid(i, permuted_train_qubo) for i in 1:get_system_size(permuted_train_qubo)]

    spins, objective = solve_mps(permuted_train_qubo, ns, 16; β=2., β_step=2, χ=2, threshold = 1e-14)

    @test spins[1] == [1, -1, 1, 1, -1, 1, 1, 1, 1]
    @test objective[1] ≈ 0.30956452652382055

    first_deg = [[1, -1, 1, 1, -1, 1, -1, 1, 1], [1, -1, 1, 1, -1, 1, 1, -1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1]]
    @test spins[2] in first_deg
    @test spins[3] in first_deg
    @test spins[4] in first_deg
    @test objective[2] ≈ 0.20750730767045347
    @test objective[3] ≈ 0.20750730767045347
    @test objective[4] ≈ 0.20750730767045347

    second_deg = [[1, -1, 1, 1, -1, 1, -1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, 1], [1, -1, 1, 1, -1, 1, 1, -1, -1]]
    @test spins[5] in second_deg
    @test spins[6] in second_deg
    @test spins[7] in second_deg
    @test objective[5] ≈ 0.1390963080303899
    @test objective[6] ≈ 0.1390963080303899
    @test objective[7] ≈ 0.1390963080303899

    @test spins[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test objective[8] ≈ 0.09323904360231824
end


@testset "MPS vs PEPS larger QUBO" begin
    function make_qubo_l()
        qubo = [(1,1) 2.8; (1,2) -0.3; (1,5) -0.2; (2,2) -2.7; (2,3) -0.255; (2,6) -0.21; (3,3) 2.6; (3,4) -0.222; (3,7) -0.213; (4,4) -2.5; (4,8) -0.2]
        qubo = vcat(qubo, [(5,5) 2.4; (5,6) -0.15; (5,9) -0.211; (6,6) -2.3; (6,7) -0.2; (6,10) -0.15; (7,7) 2.2; (7,8) -0.11; (7,11) -0.35; (8,8) -2.1; (8,12) -0.19])
        qubo = vcat(qubo, [(9,9) 2.; (9,10) -0.222; (9,13) -0.15; (10,10) -1.9; (10,11) -0.28; (10,14) -0.21; (11,11) 1.8; (11,12) -0.19; (11,15) -0.18; (12,12) -1.7; (12,16) -0.27])
        qubo = vcat(qubo, [(13,13) 1.6; (13,14) -0.32; (14,14) -1.5; (14,15) -0.19; (15,15) 1.4; (15,16) -0.21; (16,16) -1.3])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    l_qubo = make_qubo_l()

    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    β = 0.5
    β_step = 2

    println("number of β steps = ", β_step)

    ns = [Node_of_grid(i, l_qubo) for i in 1:get_system_size(l_qubo)]

    spins, _ = solve_mps(l_qubo, ns, 10; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    spins_exact, _ = solve_mps(l_qubo, ns, 10; β=β, β_step=1, χ=12, threshold = 0.)

    spins_peps, _ = solve(l_qubo, grid, 10; β = β, χ = 2, threshold = 1e-12)

    for k in 1:10
        #testing exact
        @test spins_exact[k] == spins_peps[k]
        # testing approximate
        @test spins[k] == spins_peps[k]
    end
end

@testset "MPS vs PEPS larger QUBO" begin
    function make_qubo_full()
        qubo = [(1,1) 0.1; (1,2) 1.; (1,3) 1.; (1,4) 0.2; (2,2) -0.1; (2,3) 1.0; (2,4) 0.2]
        qubo = vcat(qubo, [(3,3) 0.2; (3,4) 0.2; (4,4) -0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    l_qubo = make_qubo_full()

    β = 0.5
    β_step = 2

    println("number of β steps = ", β_step)

    ns = [Node_of_grid(i, l_qubo) for i in 1:get_system_size(l_qubo)]

    spins, _ = solve_mps(l_qubo, ns, 4; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @test spins == [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]

    M = rand([-1.,-0.5,0.,0.5,1.], 64,64)
    M = M*(M')

    q = matrix2qubo_vec(M)

    ns = [Node_of_grid(i, q) for i in 1:get_system_size(q)]

    @time s, _ = solve_mps(q, ns, 4; β=β, β_step=β_step, χ=12, threshold = 1.e-8)
    println(s[1])
    println(s[2])
    println(s[3])
    println(s[4])
end
