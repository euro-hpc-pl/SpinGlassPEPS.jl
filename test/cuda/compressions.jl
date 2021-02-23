@testset "Canonisation and Compression" begin

D = 10
Dcut = 5

d = 2
sites = 5

T = Float32

ψ = CUDA.randn(CuMPS{T}, sites, D, d)
ϕ = CUDA.randn(CuMPS{T}, sites, D, d)
χ = CUDA.randn(CuMPS{T}, sites, D, d)
Φ = CUDA.randn(CuMPS{T}, sites, D, d)

@testset "Canonisation (left)" begin
    canonise!(ψ, :left)

    is_left_normalized = true
    for i ∈ 1:length(ψ)
        A = ψ[i]
        DD = size(A, 3)

        @cutensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
        is_left_normalized *= norm(Id - cu(I(DD))) < 1e-5
    end

    @test is_left_normalized
    @test dot(ψ, ψ) ≈ 1
end

@testset "Canonisation (right)" begin
    canonise!(ϕ, :right)

    is_right_normalized = true
    for i ∈ 1:length(ϕ)
        B = ϕ[i]
        DD = size(B, 1)

        @cutensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        is_right_normalized *= norm(Id - cu(I(DD))) < 1e-5
    end

    @test is_right_normalized
    @test dot(ϕ, ϕ) ≈ 1
end

@testset "Cauchy-Schwarz inequality (after truncation)" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end

@testset "Canonisation (both)" begin
    canonise!(χ)
    @test dot(χ, χ) ≈ 1
end

@testset "Truncation (SVD, right)" begin
    truncate!(ψ, :right, Dcut)
    @test dot(ψ, ψ) ≈ 1
end

@testset "Truncation (SVD, left)" begin
    truncate!(ψ, :left, Dcut)
    @test dot(ψ, ψ) ≈ 1
end

@testset "Variational compression" begin
    Dcut = 5
    tol = 1E-4
    max_sweeps = 5

    canonise!(Φ, :right)
    @test dot(Φ, Φ) ≈ 1

    Ψ = compress(Φ, Dcut, tol, max_sweeps)

    @test dot(Ψ, Ψ) ≈ 1

    overlap = dot(Ψ, Φ)
    dist1 = 2 - 2 * real(overlap)
    dist2 = norm(Ψ)^2 + norm(Φ)^2 - 2 * real(overlap)

    @test abs(dist1 - dist2) < 1e-5
end

end
