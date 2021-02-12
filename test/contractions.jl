@testset "contractions" begin

D = 10
d = 3
sites = 5
T = ComplexF64

ψ = randn(MPS{T}, sites, D, d)
ϕ = randn(MPS{T}, sites, D, d)

Id = [I(d) for _ ∈ 1:length(ϕ)]

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
end

@testset "left environment" begin
    L = left_env(ϕ, ψ)
    @test L[end][1] ≈ dot(ϕ, ψ)  
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
    σ = (1, 2, 2, 1, 2)

    @test tensor(ψ, σ) ≈ left_env(ψ, σ)[]
end

@testset "right_env correctly contracts MPO with MPS for a given configuration" begin
    D = 10
    d = 2
    sites = 5
    T = Float64

    ψ = randn(MPS{T}, sites, D, d)
    W = randn(MPO{T}, sites, D, d)

    σ = (1, 1, 2, 1, 2)

    ϕ = MPS(T, sites)
    for (i, A) ∈ enumerate(W)
        m = σ[i]
        @cast B[x, s, y] := A[x, $m, y, s]
        ϕ[i] = B
    end

    @test dot(ψ, ϕ) ≈ right_env(ψ, W, σ)[]
end



end