using TensorOperations

function make_interactions_full()
    J_h = [(1,1) 0.; (1,2) 0.; (1,3) 2.; (1,4) 0.; (1,5) 2.; (2,2) 0.; (2,3) 0.; (2,4) 2.]
    J_h = vcat(J_h, [(2,5) 0.; (3,3) 0.; (3,4) 2.; (3,5) 0.; (4,4) 0.; (4,5) 0.; (5,5) 0.])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "axiliary, testing grouping of connections" begin
    b = scalar_prod_with_itself([ones(1,2,2), ones(2,1,2)])
    @test b == 16.0*ones(1,1)

    ints = make_interactions_full()
    # a grid
    M = [1 2; 3 4]
    #a vector of nodes, transpose for better indexing
    ns = [Node_of_grid(i, Array(transpose(M)), ints) for i in 1:maximum(M)]

    b_node, c_nodes = connections_for_mps(ns)
    @test b_node == [1,2,3]
    @test c_nodes == [[3, 2], [4], [4]]

    all_b_nodes, all_c_nodes = cluster_conncetions(b_node, c_nodes)

    @test all_b_nodes == [[1], [2], [3]]
    @test all_c_nodes == [[[3, 2]], [[4]], [[4]]]
    # 1 (B) with 3 and 2 (both C)
    # another mps 2 with 4
    # another mps 3 with 4

    ns = [Node_of_grid(i, ints) for i in 1:5]
    b_node, c_nodes = connections_for_mps(ns)
    @test b_node == [1,2,3,4]
    @test c_nodes == [[2,3,4,5], [3,4,5], [4,5], [5]]
    b_nodes_in_mpses, c_nodes_in_mpses = cluster_conncetions(b_node, c_nodes)
    @test b_nodes_in_mpses == [[1], [2], [3], [4]]
    @test c_nodes_in_mpses == [[[2, 3, 4, 5]], [[3, 4, 5]], [[4, 5]], [[5]]]
    # 1 mps 1 (B) conceted with 2,3,4 and 5 (C)
    # 2 mps 2 (B) conected with 3,4 and 5
    # ....

    grid = [1 2 3; 4 5 6; 7 8 9]
    M = ones(9,9)
    ints1 = M2interactions(M)

    ns = [Node_of_grid(i, grid, ints1) for i in 1:maximum(grid)]
    b_node, c_nodes  = connections_for_mps(ns)

    @test b_node == [1, 2, 3, 4, 5, 6, 7, 8]
    @test c_nodes == [[2, 4], [3, 5], [6], [5, 7], [6, 8], [9], [8], [9]]

    # this is only for larger elmentary cels
    # takes into account that cals can be conected by different spins
    b_node_new, c_nodes_new = split_if_differnt_spins(b_node, c_nodes, ns)
    @test b_node_new == [1, 2, 3, 4, 5, 6, 7, 8]
    @test c_nodes_new == [[2, 4], [3, 5], [6], [5, 7], [6, 8], [9], [8], [9]]

    all_b_nodes, all_c_nodes = cluster_conncetions(b_node, c_nodes)
    @test all_b_nodes == [[1, 5], [2, 6], [3, 7], [4, 8]]
    @test all_c_nodes == [[[2, 4], [6, 8]], [[3, 5], [9]], [[6], [8]], [[5, 7], [9]]]
    # 1 mps 1 (B) conected with 2 and 4 (C); and 5 (B) conected with 6 and 8
    # 2 mps 2 (B) conceted with 3 and 5; and 6 (B) conected with 9
end

@testset begin "larger tensors"

    M = [1 2;3 4]
    grid1 = Array{Array{Int}}(undef, (2,2))
    grid1[1,1] = [1 2; 5 6]
    grid1[1,2] = [3 4; 7 8]
    grid1[2,1] = [9 10; 13 14]
    grid1[2,2] = [11 12; 15 16]
    grid1 = Array{Array{Int}}(grid1)
    ints1 = M2interactions(ones(16,16))

    ns_large = [Node_of_grid(i, M, grid1, ints1) for i in 1:maximum(M)]

    b_node, c_nodes = connections_for_mps(ns_large)

    @test b_node == [1, 2, 3]
    @test c_nodes == [[2 ,3], [4], [4]]

    b_node_new, c_nodes_new = split_if_differnt_spins(b_node, c_nodes, ns_large)

    @test b_node_new == [1, 1, 2, 3]
    @test c_nodes_new == [[2], [3], [4], [4]]

    all_b_node, all_c_nodes = cluster_conncetions(b_node_new, c_nodes_new)
    @test all_b_node == [[1, 3], [1], [2]]
    @test all_c_nodes == [[[2], [4]], [[3]], [[4]]]
end

function make_interactions_case1()
    J_h = [(1,1) .5; (1,2) -0.5; (1,4) -1.5; (2,2) -1.; (2,3) -1.5; (2,5) -0.5; (3,3) 2.; (3,6) 1.5]
    J_h = vcat(J_h, [(6,6) .05; (5,6) -0.25; (6,9) -0.52; (5,5) 0.75; (4,5) 0.5; (5,8) 0.5; (4,4) 0.; (4,7) -0.01])
    J_h = vcat(J_h, [(7,7) 0.35; (7,8) 0.5; (8,8) -0.08; (8,9) -0.05; (9,9) 0.33])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
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

    #interactions matrix
    M = [1. 1. 1.; 1. 1. 0.; 1. 0. 1.]
    # construct MPS form tha matrix of interacion
    mps1 = construct_mps(M, 1., 1, 2, 1e-8)
    #mps modes 1 - left, 2 - right, 3 - physical

    @test length(mps1) == 3
    # this is B type tensor, only internal energy (± h/2)
    @test mps1[1][1,:,:] ≈ [exp(-1/2) 0.0; 0.0 exp(1/2)]
    # type C tensor input from internale enegy and interaction
    #±(h/2 + J) -- J is twice due to the symmetry of M
    @test mps1[2][1,:,:] ≈ [exp(1/2) exp(-1/2); 0.0 0.0]
    @test mps1[2][2,:,:] ≈ [0. 0.; exp(-1)*exp(-1/2) exp(1)*exp(1/2)]
    @test mps1[3][:,1,:] ≈ [exp(1/2) exp(-1/2); exp(-1)*exp(-1/2) exp(1)*exp(1/2)]

    # the same, detailed

    # changed to Vector{Node_of_grid}
    ints = M2interactions(M)
    ns = [Node_of_grid(i, ints) for i in 1:3]

    # computed mps, β = 1., β_step = 1   χ = 2, threshold = 1e-8
    mps = construct_mps(ns, 1., 1, 2, 1e-8)
    @test mps ≈ mps1




    interactions =  make_interactions_case1()

    β = 2.

    mps = initialize_mps(9)
    @test mps[1] == ones(1,1,2)
    @test mps[2] == ones(1,1,2)
    @test mps[3] == ones(1,1,2)

    # construct form mpo-mps
    ns = [Node_of_grid(i, interactions) for i in 1:9]
    #is, js = connections_for_mps(ns)
    #all_is, all_js = cluster_conncetions(is,js)
    mps = construct_mps(ns, β, 2, 4, 0.)

    # construct form psps for comparison
    grid = [1 2 3 ; 4 5 6; 7 8 9]
    ns_peps = [Node_of_grid(i, grid, interactions) for i in 1:9]
    M = Array{Union{Nothing, Array{Float64}}}(nothing, (3,3))
    k = 0
    for i in 1:3
        for j in 1:3
            k = k+1
            M[i,j] = compute_single_tensor(ns_peps[k], β)
        end
    end
    M = Matrix{Array{Float64, N} where N}(M)
    # compute probabilities by n-con
    cc = contract3x3by_ncon(M)

    v = ones(1)*mps[1][:,:,1]*mps[2][:,:,1]
    v = reshape(v, size(v,2))
    v1 = partial_spin_set(mps, [1,1])

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

    mps_a = construct_mps(ns, β, 3, 2, 1.e-12)
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


function make_interactions_case2(T::Type = Float64)
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

@testset "MPS - solving simple problem" begin

    ints = make_interactions_case2()

    grid = [1 2 3; 4 5 6; 7 8 9]

    ns = [Node_of_grid(i, ints) for i in 1:get_system_size(ints)]

    spins, _ = solve_mps(ints, ns, 2; β=2., β_step=2, χ=2, threshold = 1e-14)

    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]

    permuted_int = make_interactions_case3()

    ns = [Node_of_grid(i, permuted_int) for i in 1:get_system_size(permuted_int)]

    spins, objective = solve_mps(permuted_int, ns, 16; β=2., β_step=2, χ=2, threshold = 1e-14)

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

# it will be a test for larger cell tensors no mps when implemented
if false
@testset "on larger tensors" begin
    β = 2.

    M = [1 2;3 4]
    grid1 = Array{Array{Int}}(undef, (2,2))
    grid1[1,1] = [1 2; 4 5]
    grid1[1,2] = reshape([3; 6], (2,1))
    grid1[2,1] = reshape([7; 8], (1,2))
    grid1[2,2] = reshape([9], (1,1))
    grid1 = Array{Array{Int}}(grid1)

    q = make_interactions_case2()

    ns = [Node_of_grid(i, M, grid1, q) for i in 1:maximum(M)]

    solve_mps(q, ns, 2, β=β, β_step=2, χ = 10, threshold = 1e-12)
end
end

function make_interactions_larger()
    J_h = [(1,1) 2.8; (1,2) -0.3; (1,5) -0.2; (2,2) -2.7; (2,3) -0.255; (2,6) -0.21; (3,3) 2.6; (3,4) -0.222; (3,7) -0.213; (4,4) -2.5; (4,8) -0.2]
    J_h = vcat(J_h, [(5,5) 2.4; (5,6) -0.15; (5,9) -0.211; (6,6) -2.3; (6,7) -0.2; (6,10) -0.15; (7,7) 2.2; (7,8) -0.11; (7,11) -0.35; (8,8) -2.1; (8,12) -0.19])
    J_h = vcat(J_h, [(9,9) 2.; (9,10) -0.222; (9,13) -0.15; (10,10) -1.9; (10,11) -0.28; (10,14) -0.21; (11,11) 1.8; (11,12) -0.19; (11,15) -0.18; (12,12) -1.7; (12,16) -0.27])
    J_h = vcat(J_h, [(13,13) 1.6; (13,14) -0.32; (14,14) -1.5; (14,15) -0.19; (15,15) 1.4; (15,16) -0.21; (16,16) -1.3])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "MPS vs PEPS larger system" begin

    ints_larger = make_interactions_larger()

    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    β = 0.5
    β_step = 2

    println("number of β steps = ", β_step)

    ns = [Node_of_grid(i, ints_larger) for i in 1:get_system_size(ints_larger)]

    spins, _ = solve_mps(ints_larger, ns, 10; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    spins_exact, _ = solve_mps(ints_larger, ns, 10; β=β, β_step=1, χ=12, threshold = 0.)

    ns = [Node_of_grid(i, grid, ints_larger) for i in 1:maximum(grid)]
    spins_peps, _ = solve(ints_larger, ns, grid, 10; β = β, χ = 2, threshold = 1e-12)

    for k in 1:10
        #testing exact
        @test spins_exact[k] == spins_peps[k]
        # testing approximate
        @test spins[k] == spins_peps[k]
    end
end

function make_interactions_full()
    J_h = [(1,1) 0.1; (1,2) 1.; (1,3) 1.; (1,4) 0.2; (2,2) -0.1; (2,3) 1.0; (2,4) 0.2]
    J_h = vcat(J_h, [(3,3) 0.2; (3,4) 0.2; (4,4) -0.2])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end

@testset "MPS on full graph" begin

    ints_full = make_interactions_full()

    β = 0.5
    β_step = 2

    println("number of β steps = ", β_step)

    ns = [Node_of_grid(i, ints_full) for i in 1:get_system_size(ints_full)]

    spins, _ = solve_mps(ints_full, ns, 4; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @test spins == [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]

    #test if if just works on large graph 64 x 64
    Random.seed!(1234)
    M = rand([-1.,-0.5,0.,0.5,1.], 64,64)
    M = M*(M')
    q = M2interactions(M)
    ns = [Node_of_grid(i, q) for i in 1:get_system_size(q)]

    @time s, _ = solve_mps(q, ns, 4; β=1., β_step=1, χ=6, threshold = 1.e-6)
    @test length(s[1]) == 64
end
