@testset "arbitrary peps decomposition" begin

    @testset "helpers" begin

        qubo = make_qubo()

        grid = [1 2 3; 4 5 6; 7 8 9]
        M = make_pepsTN(grid, qubo, 1.)
        mps = set_spins_on_mps(M[1,:], [-1,1,1])
        mpo = set_spins_on_mps(M[2,:], [-1,1,1])

        mps1 = MPSxMPO(mpo, mps)

        mps_r = make_lower_mps(M, 2, 0, 0.)
        sp = compute_scalar_prod(mps_r, mps1)
        println(sp ≈ 1.0616371878660011e8)
        spp, _ = comp_marg_p_first(mps_r, M[1,:], [0,0,0])

        println(spp ≈ 6.0618722251626804e7)

        mps11 = set_spins_on_mps(M[1,:], [-1, 1, 0])
        sp = compute_scalar_prod(mps_r, mps11)
        println(sp ≈ 1.6414303747137474e7)

        ##### marginal probability implementation ####
        @test 32. == compute_scalar_prod([ones(1,2,2,1), 2*ones(2,1,2,1)], [ones(1,2,1,2), ones(2,1,1,2)])


        spp, _ = comp_marg_p_first(mps_r, M[1,:], [-1,1,0])
        println(spp ≈ sp)

        mps12 = set_spins_on_mps(M[1,:], [-1, 1, 1])
        sp1 = compute_scalar_prod(mps_r, mps12)
        println(sp1)

        mps = set_spins_on_mps(M[1,:], [-1,1,1])
        mpo = set_spins_on_mps(M[2,:], [-1,1,1])

        mps1 = MPSxMPO(mpo, mps)
        pss = comp_marg_p_last(mps1, M[3,:], [-1,0,0])
        psss = comp_marg_p_last(mps1, M[3,:], [-1,1,0])
        pssss = comp_marg_p_last(mps1, M[3,:], [-1,1,1])
        println(pss ≈ 8402.628527608993)
        println(psss)
        println(pssss)
    end

    @testset "solving" begin

        function make_qubo()
            css = -2.
            qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
            qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
            qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
            [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
        end

        train_qubo = make_qubo()

        grid = [1 2 3; 4 5 6; 7 8 9]

        ses1 = solve_arbitrary_decomposition(train_qubo, grid, 4; β = 1.)
        #first
        @test ses1[3].spins == [-1,1,-1,-1,1,-1,1,1,1]
        #ground
        @test ses1[4].spins == [1,-1,1,1,-1,1,1,1,1]
    end
end
