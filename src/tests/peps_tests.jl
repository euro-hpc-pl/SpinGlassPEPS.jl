
function make_qubo()
    qubo = [(1,1) 0.2; (1,2) 0.5; (1,4) 1.5; (2,2) 0.6; (2,3) 1.5; (2,5) 0.5; (3,3) 0.2; (3,6) -1.5]
    qubo = vcat(qubo, [(6,6) 2.2; (5,6) 0.25; (6,9) 0.52; (5,5) -0.2; (4,5) -0.5; (5,8) -0.5; (4,4) 2.2; (4,7) 0.01])
    qubo = vcat(qubo, [(7,7) -0.2; (7,8) -0.5; (8,8) 0.2; (8,9) 0.05; (9,9) 0.8])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "PEPS - axiliary functions" begin
    qubo = make_qubo()

    @test JfromQubo_el(qubo, 1,2) == 0.5
    @test JfromQubo_el(qubo, 2,1) == 0.5
    @test_throws BoundsError JfromQubo_el(qubo, 1,3)

    @test make_tensor_sizes(false, false, true, true , 2,2) == (1,1,2,2,2)
    @test make_tensor_sizes(true, false, true, true , 2,2) == (2,1,2,2,2)
    @test make_tensor_sizes(false, false, true, false , 2,2) == (1,1,2,1,2)

    # partial solution
    ps = Partial_sol{Float64}()
    @test ps.spins == []
    @test ps.objective == 1.

    ps1 = Partial_sol{Float64}([1,1], 1.)
    @test ps1.spins == [1,1]
    @test ps1.objective == 1.

    ps2 = add_spin(ps1, -1, 1.)
    @test ps2.spins == [1,1,-1]
    @test ps2.objective == 1.
end

@testset "PEPS network creation" begin


        @testset "testing itterative approximation" begin

            T1 = rand(1,2,2,1)
            T2 = rand(2,2,2,1)
            T3 = rand(2,1,2,1)

            A1, A2 = make_left_canonical(T1, T2)

            A11 = A1[:,:,1,1]
            A12 = A1[:,:,2,1]
            @test A11'*A11+A12'*A12 ≈ Matrix(I, 2, 2)

            @tensor begin
                V[i,m,k,n] := A1[i,j,k,l]*A2[j,m, n, l]
                W[i,m,k,n] := T1[i,j,k,l]*T2[j,m, n, l]
            end

            @test norm(abs.(V-W)) ≈ 0. atol=1e-14

            A2,A3 = make_left_canonical(T2, T3)

            X = A2[:,:,1,1]
            X1 = A2[:,:,2,1]
            @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

            B1,B2 = make_right_canonical(T1, T2)

            B11 = B2[:,:,1,1]
            B12 = B2[:,:,2,1]
            @test B11*B11'+B12*B12' ≈ Matrix(I, 2, 2)

            @tensor begin
                V[i,m,k,n] := B1[i,j,k,l]*B2[j,m, n, l]
                W[i,m,k,n] := T1[i,j,k,l]*T2[j,m, n, l]
            end

            @test norm(abs.(V-W)) ≈ 0. atol=1e-14

            v = vec_of_left_canonical([T1, T2, T3])
            X = v[1][:,:,1,1]
            X1 = v[1][:,:,2,1]
            @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

            X = v[2][:,:,1,1]
            X1 = v[2][:,:,2,1]
            @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

            v = vec_of_right_canonical([T1, T2, T3])
            B = v[2][:,:,1,1]
            B1 = v[2][:,:,2,1]
            @test B*B'+B1*B1' ≈ Matrix(I, 2, 2)

            B = v[3][:,:,1,1]
            B1 = v[3][:,:,2,1]
            @test B*B'+B1*B1' ≈ Matrix(I, 2, 2)

            U,R = QR_make_right_canonical(T2)
            U1,R1 = QR_make_left_canonical(T2)

            @test R[2,1] == 0.0
            @test R1[2,1] == 0.0

            B = U[:,:,1,1]
            B1 = U[:,:,2,1]
            @test B*B'+B1*B1' ≈ Matrix(I, 2,2)

            A = U1[:,:,1,1]
            A1 = U1[:,:,2,1]
            @test A'*A+A1'*A1 ≈ Matrix(I, 2,2)

            @test R_update(U,U, Matrix{Float64}(I, 2,2)) ≈ Matrix(I, 2,2)
            @test L_update(U1,U1, Matrix{Float64}(I, 2,2)) ≈ Matrix(I, 2,2)

            # approximations of PePses

            T1 = rand(1,8,4,1)
            T2 = rand(8,8,4,1)
            T3 = rand(8,1,4,1)


            mps_svd = left_canonical_approx([T1, T2, T3], 3)
            T11 = mps_svd[1]
            T12 = mps_svd[2]
            T13 = mps_svd[3]

            mps_svd_exact = left_canonical_approx([T1, T2, T3], 8)

            T1e = mps_svd_exact[1]
            T2e = mps_svd_exact[2]
            T3e = mps_svd_exact[3]

            @test size(T11) == (1,3,4,1)
            @test size(T12) == (3,3,4,1)
            @test size(T13) == (3,1,4,1)

            E = ones(1,1)
            @tensor begin
                D2[z1, z2, z3, v1, v2, v3] := T1[a,x,z1,v1]*T2[x,y,z2,v2]*T3[y,a,z3,v3]
                D12[z1, z2, z3, v1, v2, v3] := T11[a,x,z1,v1]*T12[x,y,z2,v2]*T13[y,a,z3,v3]
                D1e[z1, z2, z3, v1, v2, v3] := T1e[a,x,z1,v1]*T2e[x,y,z2,v2]*T3e[y,a,z3,v3]
            end

            @test norm(D2) ≈ norm(D12) atol = 1e-1
            @test norm(D2) ≈ norm(D1e)

            mps_svd = right_canonical_approx([T1, T2, T3], 3)
            T11 = mps_svd[1]
            T12 = mps_svd[2]
            T13 = mps_svd[3]

            @test size(T11) == (1,3,4,1)
            @test size(T12) == (3,3,4,1)
            @test size(T13) == (3,1,4,1)

            mps_svd_exact_r = right_canonical_approx([T1, T2, T3], 8)
            T1er = mps_svd_exact_r[1]
            T2er = mps_svd_exact_r[2]
            T3er = mps_svd_exact_r[3]

            @tensor begin
                D21[z1, z2, z3, v1, v2, v3] := T11[a,x,z1,v1]*T12[x,y,z2,v2]*T13[y,a,z3,v3]
                D1er[z1, z2, z3, v1, v2, v3] := T1er[a,x,z1,v1]*T2er[x,y,z2,v2]*T3er[y,a,z3,v3]
            end
            @test norm(D2) ≈ norm(D1er)
            @test norm(D2) ≈ norm(D21) atol = 1e-1

            mps_lc = left_canonical_approx([T1, T2, T3], 0)
            println("chi = 1")
            mps_anzatz = left_canonical_approx([T1, T2, T3], 1)
            v1 = compress_mps_itterativelly(mps_lc, mps_anzatz, 1, 1e-14)
            println("chi = 2")
            mps_anzatz = left_canonical_approx([T1, T2, T3], 2)
            v2 = compress_mps_itterativelly(mps_lc, mps_anzatz, 2, 1e-14)
            println("chi = 3")
            mps_anzatz = left_canonical_approx([T1, T2, T3], 3)
            v3 = compress_mps_itterativelly(mps_lc, mps_anzatz, 3, 1e-14)

            @test size(v3[2]) == (3,3,4,1)
            @test size(v2[2]) == (2,2,4,1)
            @test size(v1[2]) == (1,1,4,1)


            @tensor begin
                Dc3[z1, z2, z3, v1, v2, v3] := v3[1][a,x,z1,v1]*v3[2][x,y,z2,v2]*v3[3][y,a,z3,v3]
                Dc2[z1, z2, z3, v1, v2, v3] := v2[1][a,x,z1,v1]*v2[2][x,y,z2,v2]*v2[3][y,a,z3,v3]
                Dc1[z1, z2, z3, v1, v2, v3] := v1[1][a,x,z1,v1]*v1[2][x,y,z2,v2]*v1[3][y,a,z3,v3]
            end

            println("norms error itterative")
            println("χ = 3")
            println(norm(abs.(Dc3./D2.*norm(D2)).-1))
            println("χ = 2")
            println(norm(abs.(Dc2./D2.*norm(D2)).-1))
            println("χ = 1")
            println(norm(abs.(Dc1./D2.*norm(D2)).-1))

            println("norm error svd cut χ = 3")
            println(norm(abs.(D12./D2.-1)))

            # norm difference from the original one
            @test norm(abs.(Dc3./D2.*norm(D2)).-1) < norm(Dc2./D2.*norm(D2).+1)
            @test norm(abs.(Dc2./D2.*norm(D2)).-1) < norm(Dc1./D2.*norm(D2).+1)
            @test norm(abs.(Dc3./D2.*norm(D2)).-1) < norm(D12./D2.-1)

            @testset "compare with otner implemtation" begin

                T1 = rand(1,16,4,1)
                T2 = rand(16,16,4,1)
                T3 = rand(16,1,4,1)

                println("comparison with the state of art")

                tol = 1e-10

                println(" .....time of our computation ...., tol = ", tol)

                @time mps_lc = left_canonical_approx([T1, T2, T3], 0)

                @time mps_anzatz = left_canonical_approx([T1, T2, T3], 3)

                @time v3 = compress_mps_itterativelly(mps_lc, mps_anzatz, 3, tol)

                println("......")

                ts = [T1[:,:,:,1], T2[:,:,:,1], T3[:,:,:,1]]
                ts = [permutedims(e, [1,3,2]) for e in ts]

                mps_mps = Mps(ts, 3, 4)

                println("state of art time")
                @time mps_mps = simplify!(mps_mps, 3; tol = tol);


                println("1, delta of norms")

                println(norm(permutedims(v3[1][:,:,:,1], [1,3,2]).-mps_mps.M[1]))

                println("norm")

                println(norm(mps_mps.M[1]))


                println(".........")
                println("2, delta of norms")

                println(norm(permutedims(v3[2][:,:,:,1], [1,3,2]).-mps_mps.M[2]))

                println("norm")

                println(norm(mps_mps.M[2]))

                println(".........")
                println("3, delta of norms")

                println(norm(permutedims(v3[3][:,:,:,1], [1,3,2]).-mps_mps.M[3]))

                println("norm")

                println(norm(mps_mps.M[3]))


            end
        end

        @testset "testing marginal probabilities for various configurations" begin


            ####   conditional probability implementation

            mps = MPSxMPO([ones(1,2,2,1), 2*ones(2,1,2,1)], [ones(1,2,1,2), ones(2,1,1,2)])
            @test mps == [2*ones(1,4,1,1), 4*ones(4,1,1,1)]

            mps = MPSxMPO([ones(1,2,2,1,2), 2*ones(2,1,2,1,2)], [ones(1,2,1,2), ones(2,1,1,2)])
            @test mps == [2*ones(1,4,1,1,2), 4*ones(4,1,1,1,2)]


            b = compute_scalar_prod([ones(1,2,2,1), ones(2,1,2,1)], [ones(1,2,1,2,2), 2*ones(2,1,1,2)])
            @test b == [32.0, 32.0]

            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1), ones(2,2))
            @test a == [8.0 8.0; 8.0 8.0]
            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1,2), ones(2,2))
            @test a[:,:,1] == [8.0 8.0; 8.0 8.0]
            @test a[:,:,2] == [8.0 8.0; 8.0 8.0]
            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1), ones(2,2,2))
            @test a[:,:,1] == [8.0 8.0; 8.0 8.0]
            @test a[:,:,2] == [8.0 8.0; 8.0 8.0]

            v1 = [ones(1,2,1,2,2), ones(2,2,1,2,2), ones(2,2,1,2,2), ones(2,1,1,2,2)]
            v2 = [ones(1,2,2,1), ones(2,2,2,1), ones(2,2,2,1), ones(2,1,2,1)]
            a = conditional_probabs(v1, v2, [1,1,1])
            @test a == [0.5, 0.5]

            a = chain2point([ones(1,2,1,1,2), ones(2,1,1,1)])
            @test a == [0.5, 0.5]

            a = chain2point([reshape([1.,0.], (1,2,1,1)), ones(2,2,1,1,2), ones(2,2,1,1), ones(2,1,1,1)])
            @test a == [0.5, 0.5]

            a = chain2point([reshape([1.,0.], (1,2,1,1)), ones(2,1,1,1,2)])
            @test a == [0.5, 0.5]
        end
end


@testset "PEPS - solving simple train problem" begin
    # simplest train problem, small example in the train paper
    #two trains approaching the single segment in opposite directions


    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()


    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(train_qubo, grid, 4; β = 1., χ = 2)

    #first
    @test ses[3].spins == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test ses[4].spins == [1,-1,1,1,-1,1,1,1,1]

    # here we give a little Jii to 7,8,9 q-bits to allow there for 8 additional
    # combinations with low excitiation energies

    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) 1.75; (6,9) 0.; (5,5) -0.75; (4,5) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) -0.1; (7,8) 0.; (8,8) -0.1; (8,9) 0.; (9,9) -0.1])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    permuted_train_qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(permuted_train_qubo, grid, 16; β = 1., threshold = 0.)

    # this correspond to the ground
    for i in 9:16
        @test ses[i].spins[1:6] == [1,-1,1,1,-1,1]
    end

    # and this to 1st excited
    for i in 1:8
        @test ses[i].spins[1:6] == [-1,1,-1,-1,1,-1]
    end


    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()


    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(train_qubo, grid, 1; β = 1., χ = 2, threshold = 1e-14)

    #ses1 = solve_arbitrary_decomposition(train_qubo, grid, 4; β = 1.)

    for el in ses
        println(el.spins)
    end

    @testset "svd approximatimation in solution" begin

        ses_a = solve(permuted_train_qubo, grid, 16; β = 1., χ = 2)

        for i in 9:16
            @test ses_a[i].spins[1:6] == [1,-1,1,1,-1,1]
        end

        # and this to 1st excited
        for i in 1:8
            @test ses_a[i].spins[1:6] == [-1,1,-1,-1,1,-1]
        end
    end
end


@testset "PEPS  - solving it on BigFloat" begin
    T = BigFloat
    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) 1.75; (6,9) 0.; (5,5) -0.75; (4,5) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el{T}(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]

    # svd does not work for the BigFloat
    ses = solve(train_qubo, grid, 4; β = T(2.), χ = 2, threshold = T(1e-77))

    #first
    @test ses[3].spins == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test ses[4].spins == [1,-1,1,1,-1,1,1,1,1]


    @test typeof(ses[1].objective) == BigFloat
end

@testset "larger QUBO" begin
    function make_qubo()
        qubo = [(1,1) -1.; (1,2) 0.5; (1,5) 0.5; (2,2) 1.; (2,3) 0.5; (2,6) 0.5; (3,3) -1.0; (3,4) 0.5; (3,7) 0.5; (4,4) 1.0; (4,8) 0.5]
        qubo = vcat(qubo, [(5,5) -1.; (5,6) 0.5; (5,9) 0.5; (6,6) 1.; (6,7) 0.5; (6,10) 0.5; (7,7) -1.0; (7,8) 0.5; (7,11) 0.5; (8,8) 1.0; (8,12) 0.5])
        qubo = vcat(qubo, [(9,9) -1.; (9,10) 0.5; (9,13) 0.5; (10,10) 1.; (10,11) 0.5; (10,14) 0.5; (11,11) -1.0; (11,12) 0.5; (11,15) 0.5; (12,12) 1.0; (12,16) 0.5])
        qubo = vcat(qubo, [(13,13) -1.; (13,14) 0.5; (14,14) 1.; (14,15) 0.5; (15,15) -1.0; (15,16) 0.5; (16,16) 1.0])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()


    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    @time ses = solve(train_qubo, grid, 2; β = 3., χ = 2, threshold = 1e-11)
    @test ses[end].spins == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
end
