include("../src/compressions.jl")

@testset "Canonisation and Compression" begin

D = 10
Dcut = 5

d = 2
sites = 5

T = Float64

ψ = randn(MPS{T}, sites, D, d)
ϕ = randn(MPS{T}, sites, D, d)
χ = randn(MPS{T}, sites, D, d)
Φ = randn(MPS{T}, sites, D, d)

@testset "Canonisation (left)" begin
    canonise!(ψ, :left)
    show(ψ)
    @test is_left_normalized(ψ)
    @test dot(ψ, ψ) ≈ 1
end

@testset "Canonisation (right)" begin
    canonise!(ϕ, :right)
    show(ϕ)
    @test is_right_normalized(ϕ)
    @test dot(ϕ, ϕ) ≈ 1
end

@testset "Cauchy-Schwarz inequality (after truncation)" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end

@testset "Canonisation (both)" begin
    canonise!(χ)
    show(χ)
    @test dot(χ, χ) ≈ 1
end

@testset "Truncation (SVD, right)" begin
    truncate!(ψ, :right, Dcut)
    show(ψ)
    @test dot(ψ, ψ) ≈ 1
end

@testset "Truncation (SVD, left)" begin
    truncate!(ψ, :left, Dcut)
    show(ψ)
    @test dot(ψ, ψ) ≈ 1
end

@testset "<left|right>" begin
    ϵ = 1E-14
    ψ  = randn(MPS{T}, sites, D, d)

    l = copy(ψ)
    r = copy(l)
    canonise!(l, :left)
    canonise!(r, :right)

    @test dot(l, l) ≈ 1
    @test dot(r, r) ≈ 1

    @test abs(1 - abs(dot(l, r))) < ϵ
end

@testset "Variational compression" begin
    Dcut = 5
    tol = 1E-4
    max_sweeps = 5

    canonise!(Φ, :right)
    @test dot(Φ, Φ) ≈ 1

    Ψ = compress(Φ, Dcut, tol, max_sweeps)

    show(Ψ)
    @test dot(Ψ, Ψ) ≈ 1

    println("(Ψ, Ψ) = ", dot(Ψ, Ψ))
    println("(Φ, Φ) = ", dot(Φ, Φ))

    overlap = dot(Ψ, Φ)
    dist1 = 2 - 2 * abs(overlap)
    dist2 = norm(Ψ)^2 + norm(Φ)^2 - 2 * abs(overlap)

    @test abs(dist1 - dist2) < 1e-14

    println("(Φ, Ψ) = ", overlap)
    println("dist(Φ, Ψ)^2 = ", dist2)
end
end
