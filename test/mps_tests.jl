import SpinGlassPEPS: connections_for_mps, construct_mps
import SpinGlassPEPS: contract4probability, compute_probs
import SpinGlassPEPS: solve_mps, M2graph, graph4mps

Random.seed!(1234)

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
    @test mps1[1][1,:,:] ≈ [exp(1/2) 0.0; 0.0 exp(-1/2)]
    # type C tensor input from internale enegy and interaction
    #±(h/2 + J) -- J is twice due to the symmetry of M
    @test mps1[2][1,:,:] ≈ [exp(-1/2) 0.0; exp(1/2) 0.0]
    @test mps1[2][2,:,:] ≈ [0. exp(1)*exp(1/2); 0. exp(-1)*exp(-1/2)]
    @test mps1[3][:,:,1] ≈ [exp(-1/2) exp(1/2); exp(1)*exp(1/2) exp(-1)*exp(-1/2)]

    ####### compute probability ######

    β = 2.
    d = 2

    # construct form mpo-mps
    g =  make_interactions_case2()
    g2 = graph4mps(g)
    mps = construct_mps(g2, β, 2, 4, 0.)


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


    # approximation

    mps_a = construct_mps(g, β, 3, 2, 1.e-12)
    pp = compute_probs(mps, [1,1])

    @test prob_v/sum(prob_v) ≈ pp/sum(pp)
end


@testset "MPS vs PEPS test on an instance" begin

    g = make_interactions_case2()

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


@testset "MPS on full graph" begin

    β = 0.1
    β_step = 4

    M = rand([-1.,-0.5,0.,0.5,1.], 32,32)
    M = M*(M')

    g = M2graph(M)

    @time s, _ = solve_mps(g, 4; β=β, β_step=β_step, χ=10, threshold = 1.e-12)
    @test length(s[1]) == 32
end
