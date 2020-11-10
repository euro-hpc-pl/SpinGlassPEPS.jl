using Random
using Statistics

@testset "mps on full graphs" begin

    @testset "L = 15 testing with brute force" begin

        sols = 10
        system_size = 15

        # sampler of random symmetric matrix
        Random.seed!(1234)
        X = rand(system_size, system_size)
        M = cov(X)

        spins_brute, energies_brute = brute_force_solve(M, sols)

        # parametrising for mps
        interactions = M2interactions(M)
        #ns = [Node_of_grid(i, interactions) for i in 1:system_size]
        χ = 10
        β_step = 3
        β = 9.

        g = interactions2graph(interactions)
        spins_mps, objectives_mps = solve_mps(g, sols; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)
        energies_mps = [-v2energy(M, spins) for spins in spins_mps]
        p = sortperm(energies_mps)
        spins_mps = spins_mps[p]
        energies_mps = energies_mps[p]


        @test spins_mps[1:8] ≈ spins_brute[1:8]
        @test energies_mps[1:8] ≈ energies_brute[1:8]
    end


    @testset "L = 20 testing with brute force" begin

        sols = 15
        system_size = 20

        # sampler of random symmetric matrix
        Random.seed!(123)
        X = rand(system_size, system_size)
        M = cov(X)

        spins_brute, energies_brute = brute_force_solve(M, sols)

        # parametrising for mps
        interactions = M2interactions(M)
        g = interactions2graph(interactions)
        #ns = [Node_of_grid(i, interactions) for i in 1:system_size]
        χ = 10
        β_step = 3
        β = 9.

        spins_mps, objectives_mps = solve_mps(g, sols; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

        # sorting improve order of output that are a bit shufled
        energies_mps = [-v2energy(M, spins) for spins in spins_mps]
        p = sortperm(energies_mps)
        spins_mps = spins_mps[p]
        energies_mps = energies_mps[p]

        # energies are computed from spins
        @test spins_mps[1:2] ≈ spins_brute[1:2]
        @test energies_mps[1:2] ≈ energies_brute[1:2]
    end

    @testset "L = 64" begin
        Random.seed!(12)
        R = rand(64, 64)
        M = 0.01*R*transpose(R)
        M1 = copy(M)
        M[1,1] = M[1,2] = M[2,1] = M[2,2] = M[1,5] = M[5,1] = M[2,5] = M[5,2] = 2
        M[5,5] = M[5,6] = M[6,5] = M[1,6] = M[6,1] = M[2,6] = M[6,2] = M[6,6] = 2
        # the output expected is [1,1,x,x,1,1,x,.....]

        χ = 10
        β_step = 2
        β = 2.
        interactions = M2interactions(M)
        #ns = [Node_of_grid(i, interactions) for i in 1:64]
        g = interactions2graph(interactions)
        spins_mps, objectives_mps = solve_mps(g, 20; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

        @test length(spins_mps[1]) == 64
        @test issorted(objectives_mps, rev=true)
        # testing particular spins
        for i in 1:19
            spins = spins_mps[i]
            @test spins[1:2] == [1,1]
            @test spins[5:6] == [1,1]
            # shows that output do vary
            @test spins_mps[i] != spins_mps[i+1]
        end

        ##### another case ####
        #the solution required [1,x,x,...,x,1]
        M1[1,1] = M1[64,64] = M1[1,64] = M1[64,1] = 2.
        interactions = M2interactions(M1)
        #ns = [Node_of_grid(i, interactions) for i in 1:64]
        g = interactions2graph(interactions)
        χ = 10
        β_step = 2
        β = 2.

        spins_mps, objectives_mps = solve_mps(g, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

        @test length(spins_mps[1]) == 64
        @test issorted(objectives_mps, rev=true)
        # testing particular spins
        for i in 1:9
            spins = spins_mps[i]
            @test spins[1] == 1
            @test spins[64] == 1
            # shows that output do vary
            @test spins_mps[i] != spins_mps[i+1]
        end
    end
end


@testset "grid vs. brute force, testing wide spectrum" begin

    #removes bonds that do not fit to the grid
    function set_zero_from_outside_grid!(M::Matrix{Float64}, s1::Int, s2::Int)
        pairs = Vector{Int}[]
        for i in 1:s1*s2
            if (i%s1 > 0 && i < s1*s2)
                push!(pairs, [i, i+1])
            end
            if i <= s1*(s2-1)
                push!(pairs, [i, i+s1])
            end
        end

        for k in CartesianIndices(size(M))
            i1 = [k[1], k[2]]
            i2 = [k[2], k[1]]
            if !(i1 in pairs) && !(i2 in pairs) && (k[1] != k[2])
                M[i1...] = M[i2...] = 0.
            end
        end
    end

    sols = 10
    system_size = 24
    # sampler of random symmetric matrix
    Random.seed!(12)
    X = rand(system_size, system_size)
    M = cov(X)

    set_zero_from_outside_grid!(M, 6,4)

    interactions = M2interactions(M)
    grid = [1 2 3 4 5 6; 7 8 9 10 11 12; 13 14 15 16 17 18; 19 20 21 22 23 24]
    ns = [Node_of_grid(i, grid, interactions) for i in 1:maximum(grid)]

    χ = 12
    β = 2.

    spins_exact, objectives_exact = solve(interactions, ns, grid, sols+10; β=β, χ=0, threshold = 0.)

    spins_approx, objectives_approx = solve(interactions, ns, grid, sols+10; β=β, χ=χ, threshold = 1.e-12)
    indexing = [1 2 3; 4 5 6]
    grid1 = Array{Array{Int}}(undef, (2,3))

    grid1[1,1] = [1 2; 7 8]
    grid1[1,2] = [3 4; 9 10]
    grid1[1,3] = [5 6; 11 12]
    grid1[2,1] = [13 14; 19 20]
    grid1[2,2] = [15 16; 21 22]
    grid1[2,3] = [17 18; 23 24]
    grid1 = Array{Array{Int}}(grid1)
    ns_l = [Node_of_grid(i, indexing, grid1, interactions) for i in 1:maximum(indexing)]

    spins_l, objectives_l = solve(interactions, ns_l, indexing, sols+10; β=β, χ=χ, threshold = 1.e-12)

    β_step = 4
    g = interactions2graph(interactions)
    spins_mps, objectives_mps = solve_mps(g, sols+25; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

    # sorting according to energies improve outcome
    energies_mps = [-v2energy(M, spins) for spins in spins_mps]
    p = sortperm(energies_mps)
    spins_mps = spins_mps[p]

    spins_brute, energies_brute = brute_force_solve(M, sols)

    @test spins_exact[1:sols] == spins_approx[1:sols]
    @test spins_approx[1:sols] == spins_l[1:sols]
    @test spins_l[1:sols] == spins_mps[1:sols]
    @test spins_mps[1:sols] == spins_brute[1:sols]

end
