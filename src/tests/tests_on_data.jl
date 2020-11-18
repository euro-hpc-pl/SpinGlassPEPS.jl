using NPZ


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

        spins_exact, objective_exact = solve(g, 10; β = β, χ = 0, threshold = 0.)

        for i in 1:10
            @test v2energy(QM, spins_exact[i]) ≈ ens[i]
            @test spins_exact[i] == Int.(states[i,:])
        end

        # PEPS approx

        χ = 2
        spins_approx, objective_approx = solve(g, 10; β = β, χ = χ, threshold = 1e-10)

        for i in 1:10
            @test v2energy(QM, spins_approx[i]) ≈ ens[i]
            @test spins_approx[i] == Int.(states[i,:])
        end

        @test objective_exact ≈ objective_approx

    end

    @testset "peps multispin nodes" begin

        # forming a grid

        χ = 2

        spins_l, objective_l = solve(g, 10; β = β, χ = χ, threshold = 1e-10, node_size = (2,2))

        for i in 1:10
            @test v2energy(QM, spins_l[i]) ≈ ens[i]
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
            @test v2energy(QM, spins_mps[i]) ≈ ens[i]
            @test spins_mps[i] == Int.(states[i,:])
        end
    end
end


function Qubo2M(Q::Matrix{T}) where T <: AbstractFloat
    # TODO this needs by be specified, why 2v not v
    J = (Q - diagm(diag(Q)))/4
    v = dropdims(sum(J; dims=1); dims = 1)
    h = diagm(diag(Q)/2 + v*2)
    # convention
    - J - h
end


@testset "mpo-mps small instance of rail dispratching problem" begin

    # testing converts

    X = rand(10,10)
    X = X*X'

    J = Qubo2M(X)

    Delta = 4J + X
    @test maximum(abs.(Delta - diagm(diag(Delta)))) ≈ 0. atol = 1e-12

    # a sum over interactions
    Y = J - diagm(diag(J))
    v = dropdims(sum(Y; dims=1); dims = 1)

    b = -diag(X)
    # TODO why 4v not 2v
    @test maximum(abs.(2*diag(J) - 4*v - b)) ≈ 0. atol = 1e-12


    # tests on data
    file = "tests/data/QUBO8qbits"
    s = 8

    # reading data from txt
    data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(file))))

    # converting data
    M = zeros(s,s)
    for d in data
        i = ceil(Int, d[1])+1
        j = ceil(Int, d[2])+1
        M[i,j] = d[3]
    end

    # TODO use the read function
    J = Qubo2M(M)
    g = M2graph(J)

    # parameters of the solver
    χ = 10
    β = .25
    β_step = 1

    print("mpo mps time  =  ")

    @time spins_mps, objective_mps = solve_mps(g, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    binary_mps = [spins2binary(el) for el in spins_mps]
    enenrgies_from_binary = [transpose(x)*M*x for x in binary_mps]

    @test enenrgies_from_binary[1:4] ≈ [-3.33, -3.17, -3.17, -3.0]
    @test issorted(objective_mps, rev=true) == true

    @test binary_mps[1] == [0,1,0,0,1,0,0,0]
    # we have the degeneracy
    @test binary_mps[2] in [[0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0]]
    @test binary_mps[3] in [[0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0]]

    @test binary_mps[4] == [0, 0, 0, 1, 1, 0, 0, 0]

end
