@testset "mps on full graphs" begin

    @testset "L = 15 testing with brute force" begin

        sols = 10
        system_size = 15

        # sampler of random symmetric matrix
        Random.seed!(1234)
        X = rand(system_size, system_size)
        M = cov(X)

        χ = 15
        β_step = 2
        β = 2.

        g = M2graph(M)

        spectrum = brute_force(g; num_states=sols)

        energies_brute = spectrum.energies
        spins_brute = spectrum.states

        spins_mps, objectives_mps = solve_mps(g, sols; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

        energies_mps = [energy(spins, g) for spins in spins_mps]
        p = sortperm(energies_mps)
        spins_mps = spins_mps[p]
        energies_mps = energies_mps[p]

        @test spins_mps[1] ≈ spins_brute[1]
        @test spins_mps[2] ≈ spins_brute[2]
        @test spins_mps[3] ≈ spins_brute[3]
        @test spins_mps[4] ≈ spins_brute[4]
        @test spins_mps[5] ≈ spins_brute[5]
        @test spins_mps[6] ≈ spins_brute[6]
        @test spins_mps[7] ≈ spins_brute[7]
        @test spins_mps[8] ≈ spins_brute[8]
        @test energies_mps[1:8] ≈ energies_brute[1:8]
    end

    @testset "L = 32" begin
        #Random.seed!(12)
        M = zeros(32, 32)

        M[1,1] = M[1,2] = M[2,1] = M[2,2] = M[1,5] = M[5,1] = M[2,5] = M[5,2] = 1.
        M[5,5] = M[5,6] = M[6,5] = M[1,6] = M[6,1] = M[2,6] = M[6,2] = M[6,6] = 1.
        # the output expected is [1,1,x,x,1,1,x,.....]

        χ = 10
        β = 0.1
        β_step = 1

        g = M2graph(M)

        spins_mps, objectives_mps = solve_mps(g, 20; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)

        @test length(spins_mps[1]) == 32
        @test issorted(objectives_mps, rev=true)
        # testing particular spins
        for i in 1:19
            spins = spins_mps[i]
            @test spins[1:2] == [1,1]
            @test spins[5:6] == [1,1]
            # shows that output do vary
            @test spins_mps[i] != spins_mps[i+1]
        end

    end
end
