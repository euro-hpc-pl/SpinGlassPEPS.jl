using LowRankApprox

@testset "Truncation with standard SVD works correctly" begin
    D = 100
    Dcut = D - 1
    tol = 1E-8

    a = rand(D, D)

    U1, Σ1, V1 = svd(a)

    δ = min(Dcut, size(Σ1)...)
    U1 = U1[:, 1:δ]
    Σ1 = Σ1[1:δ]
    V1 = V1[:, 1:δ]

    U2, Σ2, V2 = svd(a)

    δ = min(Dcut, size(Σ2)...)
    U2 = U2[:, 1:δ]
    Σ2 = Σ2[1:δ]
    V2 = V2[:, 1:δ]

    r1 = U1 * Diagonal(Σ1) * V1'
    r2 = U2 * Diagonal(Σ2) * V2'

    @test norm(r1 - r2) < tol
end


# psvd is randomized, so run-to-run reproducibility cannot be asserted; what it
# must guarantee is recovery of a matrix that actually has the requested rank.
@testset "Truncation with random SVD works correctly" begin
    D, r = 100, 50
    B, C = rand(D, r), rand(r, D)
    a = B * C  # exactly rank r

    U, Σ, V = psvd(a, rank = r, atol = 1E-16, rtol = 1E-16)

    @test norm(a - U * Diagonal(Σ) * V') < 1E-8 * norm(a)
end

# The CUSOLVER path only engages above QR_GPU_MIN_ELEMENTS; small fixtures
# elsewhere in the suite take the CPU fallback, so exercise it directly here.
if CUDA.functional()
    @testset "Device-native QR/RQ above the size threshold" begin
        for T in (Float64, Float32)
            M = CUDA.rand(T, 512, 128)  # 65536 elements >= 2^15
            @test length(M) >= SpinGlassTensors.QR_GPU_MIN_ELEMENTS
            Q, R = qr_fact(M)
            @test Q isa CuMatrix{T} && R isa CuMatrix{T}
            @test isapprox(Array(Q * R), Array(M); rtol = sqrt(eps(T)))
            @test isapprox(Array(Q' * Q), I(128); atol = 100 * sqrt(eps(T)))
            # agreement with the CPU path (both sign-fixed, so comparable)
            Qc, Rc = qr_fact(Array(M); toGPU = false)
            @test isapprox(Array(Q), Qc; rtol = 100 * sqrt(eps(T)))
            @test isapprox(Array(R), Rc; rtol = 100 * sqrt(eps(T)))

            Rr, Qr = rq_fact(M)
            @test isapprox(Array(Rr * Qr), Array(M); rtol = sqrt(eps(T)))
            # truncated GPU branch (hits svd_fact on device)
            Qt, Rt = qr_fact(M, 64, T(1e-12))
            @test size(Qt, 2) == 64 && size(Rt, 1) == 64
        end
    end
end
