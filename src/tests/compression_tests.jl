
@testset "testing itterative compression" begin

    Random.seed!(21)

    T1 = rand(1,2,2)
    T2 = rand(2,2,2)
    T3 = rand(2,1,2)

    A1, A2 = make_left_canonical(T1, T2)

    A11 = A1[:,:,1]
    A12 = A1[:,:,2]
    @test A11'*A11+A12'*A12 ≈ Matrix(I, 2, 2)

    @tensor begin
        V[i,m,k,n] := A1[i,j,k]*A2[j,m, n]
        W[i,m,k,n] := T1[i,j,k]*T2[j,m, n]
    end

    @test norm(abs.(V-W)) ≈ 0. atol=1e-14

    A2,A3 = make_left_canonical(T2, T3)

    X = A2[:,:,1]
    X1 = A2[:,:,2]
    @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

    B1,B2 = make_right_canonical(T1, T2)

    B11 = B2[:,:,1]
    B12 = B2[:,:,2]
    @test B11*B11'+B12*B12' ≈ Matrix(I, 2, 2)

    @tensor begin
        V[i,m,k,n] := B1[i,j,k]*B2[j,m, n]
        W[i,m,k,n] := T1[i,j,k]*T2[j,m, n]
    end

    @test norm(abs.(V-W)) ≈ 0. atol=1e-14

    v = vec_of_left_canonical([T1, T2, T3])
    X = v[1][:,:,1]
    X1 = v[1][:,:,2]
    @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

    X = v[2][:,:,1]
    X1 = v[2][:,:,2]
    @test X'*X+X1'*X1 ≈ Matrix(I, 2, 2)

    v = vec_of_right_canonical([T1, T2, T3])
    B = v[2][:,:,1]
    B1 = v[2][:,:,2]
    @test B*B'+B1*B1' ≈ Matrix(I, 2, 2)

    B = v[3][:,:,1]
    B1 = v[3][:,:,2]
    @test B*B'+B1*B1' ≈ Matrix(I, 2, 2)

    U,R = QR_make_right_canonical(T2)
    U1,R1 = QR_make_left_canonical(T2)

    @test R[2,1] == 0.0
    @test R1[2,1] == 0.0

    B = U[:,:,1]
    B1 = U[:,:,2]
    @test B*B'+B1*B1' ≈ Matrix(I, 2,2)

    A = U1[:,:,1]
    A1 = U1[:,:,2]
    @test A'*A+A1'*A1 ≈ Matrix(I, 2,2)

    @test R_update(U,U, Matrix{Float64}(I, 2,2)) ≈ Matrix(I, 2,2)
    @test L_update(U1,U1, Matrix{Float64}(I, 2,2)) ≈ Matrix(I, 2,2)

    # approximations of PEPSes

    Random.seed!(21)

    T1 = rand(1,8,4)
    T2 = rand(8,8,4)
    T3 = rand(8,1,4)

    mps_svd = left_canonical_approx([T1, T2, T3], 3)
    T11 = mps_svd[1]
    T12 = mps_svd[2]
    T13 = mps_svd[3]

    mps_svd_exact = left_canonical_approx([T1, T2, T3], 8)

    T1e = mps_svd_exact[1]
    T2e = mps_svd_exact[2]
    T3e = mps_svd_exact[3]

    @test size(T11) == (1,3,4)
    @test size(T12) == (3,3,4)
    @test size(T13) == (3,1,4)

    E = ones(1,1)
    @tensor begin
        D2[z1, z2, z3] := T1[a,x,z1]*T2[x,y,z2]*T3[y,a,z3]
        D12[z1, z2, z3] := T11[a,x,z1]*T12[x,y,z2]*T13[y,a,z3]
        D1e[z1, z2, z3] := T1e[a,x,z1]*T2e[x,y,z2]*T3e[y,a,z3]
    end

    @test norm(D2) ≈ norm(D12) atol = 1e-1
    @test norm(D2) ≈ norm(D1e)

    mps_svd = right_canonical_approx([T1, T2, T3], 3)
    T11 = mps_svd[1]
    T12 = mps_svd[2]
    T13 = mps_svd[3]

    @test size(T11) == (1,3,4)
    @test size(T12) == (3,3,4)
    @test size(T13) == (3,1,4)

    mps_svd_exact_r = right_canonical_approx([T1, T2, T3], 8)
    T1er = mps_svd_exact_r[1]
    T2er = mps_svd_exact_r[2]
    T3er = mps_svd_exact_r[3]

    @tensor begin
        D21[z1, z2, z3] := T11[a,x,z1]*T12[x,y,z2]*T13[y,a,z3]
        D1er[z1, z2, z3] := T1er[a,x,z1]*T2er[x,y,z2]*T3er[y,a,z3]
    end
    @test norm(D2) ≈ norm(D1er)
    @test norm(D2) ≈ norm(D21) atol = 1e-1

    mps_lc = left_canonical_approx([T1, T2, T3], 0)
    println("chi = 1")
    mps_anzatz = left_canonical_approx([T1, T2, T3], 1)
    v1 = compress_mps_itterativelly(mps_lc, mps_anzatz, 1e-14)
    println("chi = 2")
    mps_anzatz = left_canonical_approx([T1, T2, T3], 2)
    v2 = compress_mps_itterativelly(mps_lc, mps_anzatz, 1e-14)
    println("chi = 3")
    mps_anzatz = left_canonical_approx([T1, T2, T3], 3)
    v3 = compress_mps_itterativelly(mps_lc, mps_anzatz, 1e-14)

    @test size(v3[2]) == (3,3,4)
    @test size(v2[2]) == (2,2,4)
    @test size(v1[2]) == (1,1,4)


    @tensor begin
        Dc3[z1, z2, z3] := v3[1][a,x,z1]*v3[2][x,y,z2]*v3[3][y,a,z3]
        Dc2[z1, z2, z3] := v2[1][a,x,z1]*v2[2][x,y,z2]*v2[3][y,a,z3]
        Dc1[z1, z2, z3] := v1[1][a,x,z1]*v1[2][x,y,z2]*v1[3][y,a,z3]
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

    @testset "compare with other implemtation" begin

        Random.seed!(21)

        T1 = rand(1,16,4)
        T2 = rand(16,16,4)
        T3 = rand(16,1,4)

        println("comparison with the state of art")

        tol = 1e-10

        println(" .....time of our computation ...., tol = ", tol)

        @time mps_lc = left_canonical_approx([T1, T2, T3], 0)

        @time mps_anzatz = left_canonical_approx([T1, T2, T3], 3)

        @time v3 = compress_mps_itterativelly(mps_lc, mps_anzatz, tol)

        println("......")

        ts = [T1, T2, T3]
        ts = [permutedims(e, [1,3,2]) for e in ts]
        try
            mps_mps = Mps(ts, 3, 4)

            println("state of art time")
            @time mps_mps = simplify!(mps_mps, 3; tol = tol);

            println("1, delta of norms")

            println(norm(permutedims(v3[1], [1,3,2]).-mps_mps.M[1]))

            println("norm")

            println(norm(mps_mps.M[1]))


            println(".........")
            println("2, delta of norms")

            println(norm(permutedims(v3[2], [1,3,2]).-mps_mps.M[2]))

            println("norm")

            println(norm(mps_mps.M[2]))

            println(".........")
            println("3, delta of norms")

            println(norm(permutedims(v3[3], [1,3,2]).-mps_mps.M[3]))

            println("norm")

            println(norm(mps_mps.M[3]))
        catch
            println("no MPStates.jl for comparison of the approximation")
        end
    end
end
