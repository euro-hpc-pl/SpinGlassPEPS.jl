using NPZ


# TODO this needs by be specified, why 2v not v

function M2qubo(Q::Matrix{T}) where T <: AbstractFloat
    J = (Q - diagm(diag(Q)))/4
    v = dropdims(sum(J; dims=1); dims = 1)
    h = diagm(diag(Q)/2 + v*2)
    J + h
end

function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)

    transpose(v)*M*v + transpose(v)*d
end


function M2interactions(M::Matrix{Float64})
    qubo = Qubo_el{Float64}[]
    s = size(M)
    for i in 1:s[1]
        for j in i:s[2]
            if (M[i,j] != 0.) | (i == j)
                x = M[i,j]
                q = Qubo_el{Float64}((i,j), x)
                push!(qubo, q)
            end
        end
    end
    qubo
end


@testset "grid including larger bloks" begin

    β = 3.
    file = "example4tests.npz"
    data = npzread("./tests/data/"*file)
    println(file)

    QM = data["Js"]
    ens = data["energies"]
    states = data["states"]

    qubo = M2interactions(QM)


    grid = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    # PEPS exact

    spins, objective = solve(qubo, ns, grid, 10; β = β, χ = 0, threshold = 0.)

    for i in 1:10
        @test v2energy(QM, spins[i]) ≈ -ens[i]
        @test spins[i] == Int.(states[i,:])
    end

    # PEPS approx

    χ = 2
    spins_a, objective_a = solve(qubo, ns, grid, 10; β = β, χ = χ, threshold = 1e-10)

    for i in 1:10
        @test v2energy(QM, spins_a[i]) ≈ -ens[i]
        @test spins_a[i] == Int.(states[i,:])
    end

    @test objective ≈ objective_a


    # PEPS on blocks

    # forming a grid

    M = [1 2 3; 4 5 6; 7 8 9]
    grid1 = Array{Array{Int}}(undef, (3,3))
    grid1[1,1] = [1 2; 6 7]
    grid1[1,2] = [3 4; 8 9]
    grid1[1,3] = reshape([5; 10], (2,1))
    grid1[2,1] = [11 12; 16 17]
    grid1[2,2] = [13 14; 18 19]
    grid1[2,3] = reshape([15; 20], (2,1))

    grid1[3,1] = reshape([21; 22], (1,2))
    grid1[3,2] = reshape([23; 24], (1,2))
    grid1[3,3] = reshape([25], (1,1))

    grid1 = Array{Array{Int}}(grid1)

    ns_l = [Node_of_grid(i, M, grid1) for i in 1:maximum(M)]

    χ = 2

    spins_l, objective_l = solve(qubo, ns_l, M, 10; β = β, χ = χ, threshold = 1e-10)

    for i in 1:10
        @test v2energy(QM, spins_l[i]) ≈ -ens[i]
        @test spins_l[i] == Int.(states[i,:])
    end

    @test objective ≈ objective_l

    # MPS MPO treates as a graph without the structure
    ns = [Node_of_grid(i, qubo) for i in 1:get_system_size(qubo)]

    χ = 10
    β_step = 2

    spins_mps, objective_mps = solve_mps(qubo, ns, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-8)

    for i in 1:10
        @test v2energy(QM, spins_mps[i]) ≈ -ens[i]
        @test spins_mps[i] == Int.(states[i,:])
    end
end

@testset "mps full graph real problem" begin

    X = rand(10,10)
    X = X*X'

    J = M2qubo(X)

    Delta = 4J - X
    @test maximum(abs.(Delta - diagm(diag(Delta)))) ≈ 0. atol = 1e-12

    # a sum over interactions
    Y = J - diagm(diag(J))
    v = dropdims(sum(Y; dims=1); dims = 1)

    b = diag(X)
    # TODO why 4v not 2v
    @test maximum(abs.(2*diag(J) - 4*v - b)) ≈ 0. atol = 1e-12


    file = "tests/data/QUBO8qbits"
    s = 8
    #file = "tests/data/QUBO6qbits"
    #s = 6

    data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(file))))

    M = zeros(s,s)
    for d in data
        i = ceil(Int, d[1])+1
        j = ceil(Int, d[2])+1
        M[i,j] = d[3]
    end

    #-1 this for the model
    J = -1*M2qubo(M)
    q_vec = M2interactions(J)

    χ = 10
    β = .5
    β_step = 2

    print("mpo mps time  =  ")
    ns = [Node_of_grid(i, q_vec) for i in 1:get_system_size(q_vec)]
    @time spins_mps, objective_mps = solve_mps(q_vec, ns, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    println("first 10 states")
    for spin in spins_mps
        #println(v2energy(J, spin))
        binary = [Int(i > 0) for i in spin]
        println(binary)
        println(binary'*M*binary)
    end


    if file == "tests/data/QUBO8qbits"
        println("testing ground")
        binary = [Int(i > 0) for i in spins_mps[1]]
        @test binary == [0,1,0,0,1,0,0,0]
    elseif file == "tests/data/QUBO6qbits"
        println("testing ground")
        binary = [Int(i > 0) for i in spins_mps[1]]
        @test binary == [0,1,0,1,0,0]
    end
end
