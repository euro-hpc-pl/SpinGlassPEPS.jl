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
