using TensorOperations

if false
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

function contract3x3by_ncon(M::Matrix{Array{T, 5}}) where T <: AbstractFloat
    u1 = M[1,1][1,:,1,:,:]
    v1 = [2,31, -1]

    u2 = M[1,2][:,:,1,:,:]
    v2 = [2,3,32,-2]

    u3 = M[1,3][:,1,1,:,:]
    v3 = [3,33,-3]

    m1 = M[2,1][1,:,:,:,:]

    v4 = [4,  31, 41, -4]
    m2 = M[2,2]

    v5 = [4, 5, 32, 42, -5]
    m3 = M[2,3][:,1,:,:,:]

    v6 = [5, 33, 43, -6]

    d1 = M[3,1][1,:,:,1,:]

    v7 = [6, 41, -7]
    d2 = M[3,2][:,:,:,1,:]

    v8 = [6,7,42,-8]
    d3 = M[3,3][:,1,:,1,:]

    v9 = [7, 43, -9]

    tensors = (u1, u2, u3, m1, m2, m3, d1, d2, d3)
    indexes = (v1, v2, v3, v4, v5, v6, v7, v8, v9)

    ncon(tensors, indexes)
end


@testset "peps computing" begin

    qubo =  make_qubo_x()

    β = 2.

    mps = initialize_mps(9)
    @test mps[1] == ones(1,1,2)
    @test mps[2] == ones(1,1,2)
    @test mps[3] == ones(1,1,2)

    all_is = [[1,9], [3], [7], [5]]
    all_js = [[[2,4],[6,8]], [[2,6]], [[4,8]], [[2,4,6,8]]]

    mps = construct_mps(qubo, β, 2, 9, all_is, all_js, 4, 0.)

    grid = [1 2 3 ; 4 5 6; 7 8 9]
    M = make_pepsTN(grid, qubo, β)
    cc = contract3x3by_ncon(M)

    v = ones(1)*mps[1][:,:,1]*mps[2][:,:,1]
    v = reshape(v, size(v,2))
    v1 = v_from_mps(mps, [-1,-1])

    @test v == v1

    A = mps[3]
    B = zeros(2,2)
    M = compute_scalar_prod(mps[4:end],mps[4:end])
    @tensor begin
        B[x,y] = A[a,b,x]*A[c,d,y]*v1[a]*v1[c]*M[b,d]
    end

    @test compute_probs(mps, [-1,-1]) ≈ diag(B)

    @test compute_probs(mps, [-1,-1]) ≈ sum(cc[1,1,:,:,:,:,:,:,:], dims = (2,3,4,5,6,7))
    @test compute_probs(mps, [-1,-1,1,1,-1]) ≈ sum(cc[1,1,2,2,1,:,:,:,:], dims = (2,3,4))
    @test compute_probs(mps, [-1,-1,1,1,-1,1,1,-1]) ≈ cc[1,1,2,2,1,2,2,1,:]

    # approximation

    mps_a = construct_mps(qubo, β, 3, 9, all_is, all_js, 2, 1.e-12)
    ps = sum(cc[1,1,:,:,:,:,:,:,:], dims = (2,3,4,5,6,7))
    pp = compute_probs(mps, [-1,-1])

    @test ps/sum(ps) ≈ pp/sum(pp)

    ps = sum(cc[1,1,2,2,:,:,:,:,:], dims = (2,3,4,5))
    pp = compute_probs(mps, [-1,-1,1,1])

    @test ps/sum(ps) ≈ pp/sum(pp)

    ps = cc[1,1,2,2,1,2,1,2,:]
    pp = compute_probs(mps, [-1,-1,1,1,-1,1,-1,1])

    @test ps/sum(ps) ≈ pp/sum(pp)
end


function make_qubo(T::Type = Float64)
    css = 2.
    qubo = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
    qubo = vcat(qubo, [(6,6) 0.; (6,5) -1.75; (6,9) 0.; (5,5) 0.75; (5,4) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
    [Qubo_el{T}(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "PEPS - solving simple train problem" begin

    train_qubo = make_qubo()


    all_is = [[1,9], [3], [7], [5]]
    all_js = [[[2,4],[6,8]], [[2,6]], [[4,8]], [[2,4,6,8]]]

    ses = solve_mps(train_qubo, all_is, all_js, 9, 2; β=2., β_step=2, χ=2, threshold = 1e-14)

    #first
    @test ses[2].spins == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test ses[1].spins == [1,-1,1,1,-1,1,1,1,1]

    # here we give a little Jii to 7,8,9 q-bits to allow there for 8 additional
    # combinations with low excitiation energies

    function make_qubo1()
        css = 2.
        qubo = [(1,1) 1.25; (1,2) -1.75; (1,4) css; (2,2) 1.75; (2,3) -1.75; (2,5) 0.; (3,3) 1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) -1.75; (6,9) 0.; (5,5) 0.75; (4,5) -1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.1; (7,8) 0.; (8,8) 0.1; (8,9) 0.; (9,9) 0.1])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    permuted_train_qubo = make_qubo1()

    @time ses = solve_mps(permuted_train_qubo, all_is, all_js, 9, 16; β=2., β_step=2, χ=2, threshold = 1e-14)

    # this correspond to the ground
    for i in 1:8
        @test ses[i].spins[1:6] == [1,-1,1,1,-1,1]
    end

    # and this to 1st excited
    for i in 9:16
        @test ses[i].spins[1:6] == [-1,1,-1,-1,1,-1]
    end
end
end

@testset "larger QUBO" begin
    function make_qubo_l()
        qubo = [(1,1) 2.8; (1,2) -0.3; (1,5) -0.2; (2,2) -2.7; (2,3) -0.255; (2,6) -0.21; (3,3) 2.6; (3,4) -0.222; (3,7) -0.213; (4,4) -2.5; (4,8) -0.2]
        qubo = vcat(qubo, [(5,5) 2.4; (5,6) -0.15; (5,9) -0.211; (6,6) -2.3; (6,7) -0.2; (6,10) -0.15; (7,7) 2.2; (7,8) -0.11; (7,11) -0.35; (8,8) -2.1; (8,12) -0.19])
        qubo = vcat(qubo, [(9,9) 2.; (9,10) -0.222; (9,13) -0.15; (10,10) -1.9; (10,11) -0.28; (10,14) -0.21; (11,11) 1.8; (11,12) -0.19; (11,15) -0.18; (12,12) -1.7; (12,16) -0.27])
        qubo = vcat(qubo, [(13,13) 1.6; (13,14) -0.32; (14,14) -1.5; (14,15) -0.19; (15,15) 1.4; (15,16) -0.21; (16,16) -1.3])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    l_qubo = make_qubo_l()

    all_is = [[1, 10], [3, 12], [6, 13], [11], [5,15], [8]]
    all_js = [[[2,5], [6,9,11,14]], [[2,4,7], [8,11,16]], [[2,5,7], [9,14]], [[7, 15]], [[9],[14,16]], [[4,7]]]

    β = 1.5
    β_step = 6

    println("step = ", β_step)

    @time ses = solve_mps(l_qubo, all_is, all_js, 16, 10; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @time ses_exact = solve_mps(l_qubo, all_is, all_js, 16, 10; β=β, β_step=1, χ=12, threshold = 0.)


    @test ses[1].spins == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    @time ses_pepes = solve(l_qubo, grid, 10; β = β, χ = 2, threshold = 1e-12)

    @test ses[1].spins == ses_pepes[1].spins

    for k in 1:10
        println(ses[k].objective, ", ", ses_exact[k].objective, ", ", ses_pepes[k].objective)
        aa = (ses[k].spins == ses_pepes[k].spins)
        println(aa)
        println("exact ", ses_exact[k].spins == ses_pepes[k].spins)
        if !aa
            println(ses[k].spins)
            println(ses_pepes[k].spins)
        end
    end


end
