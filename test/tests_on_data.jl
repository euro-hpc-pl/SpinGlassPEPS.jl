@testset "mpo-mps small instance of rail dispratching problem" begin

    # the matrix made from the disparching problem
    JJ = [-2.625 -0.4375 -0.4375 -0.4375 -0.4375 0.0 0.0 0.0;
    -0.4375 -2.71 -0.4375 -0.4375 0.0 -0.4375 0.0 0.0;
    -0.4375 -0.4375 -2.79 -0.4375 0.0 0.0 -0.4375 0.0;
    -0.4375 -0.4375 -0.4375 -2.875 0.0 0.0 0.0 -0.4375;
    -0.4375 0.0 0.0 0.0 -2.625 -0.4375 -0.4375 -0.4375;
    0.0 -0.4375 0.0 0.0 -0.4375 -2.79 -0.4375 -0.4375;
    0.0 0.0 -0.4375 0.0 -0.4375 -0.4375 -2.96 -0.4375;
    0.0 0.0 0.0 -0.4375 -0.4375 -0.4375 -0.4375 -3.125]


    g = M2graph(JJ, -1)

    # parameters of the solver
    χ = 10
    β = .25
    β_step = 2

    print("mpo mps time  =  ")

    @time spins_mps, objective_mps = solve_mps(g, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    e = [energy(spins, g) for spins in spins_mps[1:4]]
    @test (e.-e[1]) ≈ [0.0, 0.16, 0.16, 0.33]

    @test issorted(objective_mps, rev=true) == true

    @test spins_mps[1] == [-1,1,-1,-1,1,-1,-1,-1]
    # we have the degeneracy
    @test spins_mps[2] in [[-1, -1, 1, -1, 1, -1, -1, -1], [1, -1, -1, -1, -1, 1, -1, -1]]
    @test spins_mps[3] in [[-1, -1, 1, -1, 1, -1, -1, -1], [1, -1, -1, -1, -1, 1, -1, -1]]

    @test spins_mps[4] == [-1, -1, -1, 1, 1, -1, -1, -1]

end
