@testset "contractions" begin

D = 10
d = 3
sites = 5
T = ComplexF64

ψ = randn(MPS{T}, sites, D, d)
ϕ = randn(MPS{T}, sites, D, d)
mpo = randn(MPO{T}, 2, 2, 2)
println("......")
println(mpo)

Id = [I(d) for _ ∈ 1:length(ϕ)]

Id_m = MPO([ones(1,1,1,d) for _ ∈ 1:length(ϕ)])

@testset "dot products" begin
    @test dot(ψ, ψ) ≈ dot(ψ, ψ)
    @test dot(ψ, ϕ) ≈ conj(dot(ϕ, ψ))

    @test dot(ψ, Id, ϕ) ≈ conj(dot(ϕ, Id, ψ))
    @test dot(ψ, Id, ϕ) ≈ dot(ψ, ϕ)

    @test norm(ψ) ≈ sqrt(abs(dot(ψ, ψ)))

    ψ[end] *= 1/norm(ψ)
    @test dot(ψ, ψ) ≈ 1

    ϕ[1] *= 1/norm(ϕ)
    @test dot(ϕ, ϕ) ≈ 1

    # dot on mpo
    mpo1 = dot(mpo, mpo)
    A = mpo[1]

    @test size(mpo1[1]) == (1, 2, 4, 2)
    @test size(mpo1[2]) == (4, 2, 1, 2)
    @reduce B[(a,b), c, (d,e), f] := sum(x) A[a,c,d,x] * A[b,x,e,f]
    @test mpo1[1] ≈ B
end

@testset "left environment" begin
    L = left_env(ϕ, ψ)
    @test L[end][1] ≈ dot(ϕ, ψ)
end

@testset "right environment" begin
    R = right_env(ϕ, ψ)
    @test R[1][end] ≈ dot(ϕ, ψ)
end

@testset "Cauchy-Schwarz inequality" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end


@testset "left_env correctly contracts MPS for a given configuration" begin
    D = 10
    d = 2
    sites = 5
    T = ComplexF64

    ψ = randn(MPS{T}, sites, D, d)
    σ = 2 * (rand(sites) .< 0.5) .- 1

    @test tensor(ψ, σ) ≈ left_env(ψ, map(idx, σ))[]
end

@testset "right_env correctly contracts MPO with MPS for a given configuration" begin
    D = 10
    d = 2
    sites = 5
    T = Float64

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    σ = 2 * (rand(sites) .< 0.5) .- 1

    ϕ = MPS(T, sites)
    for (i, A) ∈ enumerate(W)
        m = idx(σ[i])
        @cast B[x, s, y] := A[x, $m, y, s]
        ϕ[i] = B
    end

    @test dot(ψ, ϕ) ≈ right_env(ψ, W, map(idx, σ))[]
end



end
