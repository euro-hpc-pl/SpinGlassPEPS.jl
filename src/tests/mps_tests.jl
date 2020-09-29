using TensorOperations

function make_qubo_x()
    qubo = [(1,1) 0.; (1,2) -0.5; (1,4) -1.5; (2,2) 0.; (2,3) -1.5; (2,5) -0.5; (3,3) 0.; (3,6) 1.5]
    qubo = vcat(qubo, [(6,6) 0.; (5,6) -0.25; (6,9) -0.52; (5,5) 0.; (4,5) 0.5; (5,8) 0.5; (4,4) 0.; (4,7) -0.01])
    qubo = vcat(qubo, [(7,7) 0.; (7,8) 0.5; (8,8) 0.; (8,9) -0.05; (9,9) 0.])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

function make_qubo_0()
    qubo = [(1,1) 0.; (1,2) -.5; (1,4) -1.5; (2,2) 0.; (2,3) -1.5; (2,5) -0.5; (3,3) 0.; (3,6) 1.5]
    qubo = vcat(qubo, [(6,6) 0.; (5,6) -0.25; (6,9) 0.; (5,5) 0.; (4,5) 0.5; (5,8) 0.5; (4,4) 0.; (4,7) 0.])
    qubo = vcat(qubo, [(7,7) 0.; (7,8) 0.0; (8,8) 0.; (8,9) 0.; (9,9) 0.])
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

@testset "initializing" begin
    mps = initialize_mps(9)
    @test mps[1] == ones(1,1,2)
    @test mps[2] == ones(1,1,2)
    @test mps[3] == ones(1,1,2)

    qubo =  make_qubo_circ()

    β = 1.

    mpo = [make_ones() for _ in 1:9]
    add_MPO!(mpo, 1, [2,4] ,qubo, β)
    add_MPO!(mpo, 9, [6,8] ,qubo, β)
    reduce_first_and_last!(mpo)

    x = MPSxMPO(mps, mpo)

    mpo = [make_ones() for _ in 1:9]
    add_MPO!(mpo, 3, [2,6] ,qubo, β)
    add_MPO!(mpo, 7, [4,8] ,qubo, β)
    reduce_first_and_last!(mpo)

    x = MPSxMPO(x, mpo)

    mpo = [make_ones() for _ in 1:9]
    add_MPO!(mpo, 5, [2,4,6,8] ,qubo, β)
    reduce_first_and_last!(mpo)
    x = MPSxMPO(x, mpo)

    println([size(x[i]) for i in 1:9])

    grid = [1 2 3 ; 4 5 6; 7 8 9]
    M = make_pepsTN(grid, qubo, β)
    cc = contract3x3by_ncon(M)

    println(sum(cc[1,1,:,:,:,:,:,:,:])/sum(cc))
    println(sum(cc[1,2,:,:,:,:,:,:,:])/sum(cc))
    println(sum(cc[2,1,:,:,:,:,:,:,:])/sum(cc))
    println(sum(cc[2,2,:,:,:,:,:,:,:])/sum(cc))

    B1 = x[1]
    C1 = x[2][:,:,1:1]
    B2 = x[1][:,:,2:2]
    C2 = x[2][:,:,2:2]

    bb = x[1][:,:,1]
    cc = x[2][:,:,1]


    Z = compute_scalar_prod(x,x)[1]
    println(Z)


    X = compute_scalar_prod(x[5:end],x[5:end])

    println(size(X))
    println(X)

    #println(Z1)
    #println(Z1./Z)

    Z1 = compute_scalar_prod([B1, C2, x[3:end]...],[B1, C2, x[3:end]...])
    println(Z1./Z)

    #Z1 = compute_scalar_prod([B2, C1, x[3:end]...],[B2, C1, x[3:end]...])
    #println(Z1./Z)

    #Z1 = compute_scalar_prod([B2, C2, x[3:end]...],[B2, C2, x[3:end]...])
    #println(Z1./Z)


end
