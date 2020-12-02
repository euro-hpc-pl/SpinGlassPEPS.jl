using TensorOperations
using LightGraphs


# will be done in graphs

@testset "grouping of connections" begin
    M = ones(5,5)
    g = M2graph(M)

    v = connections_for_mps(g)
    E = LightGraphs.SimpleGraphs.SimpleEdge
    e1 = [E(1, 2), E(1, 3), E(1, 4), E(1, 5)]
    e2 = [E(2, 3), E(2, 4), E(2, 5)]
    e3 = [E(3, 4), E(3, 5)]
    e4 = [E(4, 5)]

    @test v == [e1, e2, e3, e4]

end


@testset "MPS computing" begin

    #interactions matrix
    M = [1. 1. 1.; 1. 1. 0.; 1. 0. 1.]
    g = M2graph(M)
    g = graph4mps(g)
    # construct MPS form tha graph of interacion
    mps1 = construct_mps(g, 1., 1, 2, 1e-8)
    #mps modes 1 - left, 3 - right, 2 - physical

    @test length(mps1) == 3
    # this is B type tensor, only internal energy (± h/2)
    @test mps1[1][1,:,:] ≈ [exp(-1/2) 0.0; 0.0 exp(1/2)]
    # type C tensor input from internale enegy and interaction
    #±(h/2 + J) -- J is twice due to the symmetry of M
    @test mps1[2][1,:,:] ≈ [exp(1/2) 0.0; exp(-1/2) 0.0]
    @test mps1[2][2,:,:] ≈ [0. exp(-1)*exp(-1/2); 0. exp(1)*exp(1/2)]
    @test mps1[3][:,:,1] ≈ [exp(1/2) exp(-1/2); exp(-1)*exp(-1/2) exp(1)*exp(1/2)]

    ####### compute probability ######

    β = 2.
    d = 2

    # construct form mpo-mps
    g =  make_interactions_case1()
    g2 = graph4mps(g)
    mps = construct_mps(g2, β, 2, 4, 0.)

    # PEPS for comparison
    g1 = graph4peps(g, (1,1))
    M = form_peps(g1, β)
    # compute probabilities by n-con
    cc = contract3x3by_ncon(M)


    v = ones(1)*mps[1][:,1,:]*mps[2][:,1,:]
    v = reshape(v, size(v,2))

    A = mps[3]
    B = zeros(2,2)
    mps1 = MPS([mps[i] for i in 4:length(mps)])
    M = right_env(mps, mps)[4]
    @tensor begin
        B[x,y] = A[a,x,b]*A[c,y,d]*v[a]*v[c]*M[b,d]
    end
    prob_v = diag(B)

    @test contract4probability(A,M,v) ≈ prob_v

    @test compute_probs(mps, [1,1]) ≈ prob_v


    @test compute_probs(mps, [1,1]) ≈ sum(cc[1,1,:,:,:,:,:,:,:], dims = (2,3,4,5,6,7))
    @test compute_probs(mps, [1,1,2,2,1]) ≈ sum(cc[1,1,2,2,1,:,:,:,:], dims = (2,3,4))
    @test compute_probs(mps, [1,1,2,2,1,2,2,1]) ≈ cc[1,1,2,2,1,2,2,1,:]

    # approximation

    mps_a = construct_mps(g, β, 3, 2, 1.e-12)
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



@testset "MPS - solving simple problem" begin

    g = interactions_case2()

    spins, _ = solve_mps(g, 2; β=2., β_step=2, χ=2, threshold = 1e-14)

    #ground
    @test spins[1] == [1,-1,1,1,-1,1,1,1,1]

    #first
    @test spins[2] == [-1,1,-1,-1,1,-1,1,1,1]

    # introduce degeneracy
    set_prop!(g, 7, :h, 0.2)
    set_prop!(g, 8, :h, 0.2)
    set_prop!(g, 9, :h, 0.2)

    spins, objective = solve_mps(g, 16; β=2., β_step=2, χ=2, threshold = 1e-14)

    @test spins[1] == [1, -1, 1, 1, -1, 1, 1, 1, 1]
    @test objective[1] ≈ 1.5332831372810425

    first_deg = [[1, -1, 1, 1, -1, 1, -1, 1, 1], [1, -1, 1, 1, -1, 1, 1, -1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1]]
    @test spins[2] in first_deg
    @test spins[3] in first_deg
    @test spins[4] in first_deg
    @test objective[2] ≈ 0.6889485237728951
    @test objective[3] ≈ 0.6889485237728951
    @test objective[4] ≈ 0.6889485237728951

    second_deg = [[1, -1, 1, 1, -1, 1, -1, 1, -1], [1, -1, 1, 1, -1, 1, -1, -1, 1], [1, -1, 1, 1, -1, 1, 1, -1, -1]]
    @test spins[5] in second_deg
    @test spins[6] in second_deg
    @test spins[7] in second_deg
    @test objective[5] ≈ 0.3095645265169637
    @test objective[6] ≈ 0.3095645265169637
    @test objective[7] ≈ 0.3095645265169637

    @test spins[8] == [1, -1, 1, 1, -1, 1, -1, -1, -1]
    @test objective[8] ≈ 0.13909630802730527
end


@testset "MPS vs PEPS larger system" begin

    g = make_interactions_large()

    β = 0.5
    β_step = 2

    println("number of β steps = ", β_step)

    spins, _ = solve_mps(g, 10; β=β, β_step=β_step, χ=12, threshold = 1.e-8)

    @test spins[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    spins_exact, _ = solve_mps(g, 10; β=β, β_step=1, χ=12, threshold = 0.)

    spins_peps, _ = solve(g, 10; β = β, χ = 2, threshold = 1e-12)

    for k in 1:10
        #testing exact
        @test spins_exact[k] == spins_peps[k]
        # testing approximate
        @test spins[k] == spins_peps[k]
    end
end


function make_interactions_full()
    J_h = [1 1 -0.1; 1 2 -1.; 1 3 -1.; 1 4 -0.2; 2 2 0.1; 2 3 -1.0; 2 4 -0.2]
    J_h = vcat(J_h, [3 3 -0.2; 3 4 -0.2; 4 4 0.2])

    L = 4
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
    0
end

@testset "MPS on full graph" begin

    L = 4
    D = Dict((1, 1) => -0.1, (1, 2) => -1., (1, 3) => -1., (1, 4) => -0.2, (2, 2) => 0.1)
    D1 = Dict((2, 3) => -1.0, (2, 4) => -0.2, (3, 3) => -0.2, (3, 4) => -0.2, (4, 4) => 0.2)
    D2 = merge!(+, D, D1)
    g = ising_graph(D2, L, 1, -1)

    spins, _ = solve_mps(g, 4; β=1., β_step=2, χ=12, threshold = 1.e-8)

    @test spins[1] in [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]
    @test spins[2] in [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]
    @test spins[3] in [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]
    @test spins[4] in [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]

    β = 0.1
    β_step = 4

    #test if works on large graph 64 x 64
    Random.seed!(1234)
    M = rand([-1.,-0.5,0.,0.5,1.], 64,64)
    M = M*(M')

    g = M2graph(M)

    @time s, _ = solve_mps(g, 4; β=β, β_step=β_step, χ=25, threshold = 1.e-12)
    @test length(s[1]) == 64
end
