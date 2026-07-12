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


@testset "Truncation with with random SVD works correctly" begin

    D = 100
    Dcut = D - 1
    tol = 1E-8

    a = rand(D, D)

    U1, Σ1, V1 = psvd(a, rank = Dcut, atol = 1E-16, rtol = 1E-16)
    U2, Σ2, V2 = psvd(a, rank = Dcut, atol = 1E-16, rtol = 1E-16)

    r1 = U1 * Diagonal(Σ1) * V1'
    r2 = U2 * Diagonal(Σ2) * V2'

    println(norm(a - r2))
    println(norm(a - r1))

    @test norm(r1 - r2) < tol
end
