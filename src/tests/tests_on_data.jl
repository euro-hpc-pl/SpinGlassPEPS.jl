@testset "testing peps and mpo-mps on grid, random instance" begin

    β = 3.
    file = "example4tests.npz"
    data = npzread("./tests/data/"*file)
    println(file)

    QM = data["Js"]
    ens = data["energies"]
    states = data["states"]

    g = M2graph(QM)

    objective_exact = 0.

    @testset "peps exact and approximated one spin nodes" begin

        spins_exact, objective_exact = solve(g, 10; β = β, threshold = 0.)

        for i in 1:10
            @test energy(spins_exact[i], g) ≈ ens[i]
            @test spins_exact[i] == Int.(states[i,:])
        end

        # PEPS approx

        χ = 2
        spins_approx, objective_approx = solve(g, 10; β = β, χ = χ, threshold = 1e-10)

        for i in 1:10
            @test energy(spins_approx[i], g) ≈ ens[i]
            @test spins_approx[i] == Int.(states[i,:])
        end

        @test objective_exact ≈ objective_approx

    end

    @testset "peps multispin nodes" begin

        # forming a grid

        χ = 2

        spins_l, objective_l = solve(g, 10; β = β, χ = χ, threshold = 1e-10, node_size = (2,2))

        for i in 1:10
            @test energy(spins_l[i], g) ≈ ens[i]
            @test spins_l[i] == Int.(states[i,:])
        end

        @test objective_exact ≈ objective_l

    end

    @testset "mpo-mps one spin nodes" begin

        # MPS MPO treates as a graph without the structure

        χ = 15
        β_step = 2

        spins_mps, objective_mps = solve_mps(g, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-8)

        for i in 1:10
            @test energy(spins_mps[i], g) ≈ ens[i]
            @test spins_mps[i] == Int.(states[i,:])
        end
    end
end

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


    g = M2graph(JJ)

    # parameters of the solver
    χ = 10
    β = .25
    β_step = 1

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
