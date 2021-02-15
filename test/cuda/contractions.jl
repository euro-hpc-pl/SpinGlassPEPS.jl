@testset "contractions" begin

D = 10
d = 3
sites = 5
T = Float32

ψ = CUDA.randn(CuMPS{T}, sites, D, d)
ϕ = CUDA.randn(CuMPS{T}, sites, D, d)

Id = [CuMatrix{T}(I(d)) for _ ∈ 1:length(ϕ)]

@testset "dot products" begin
    @test dot(ψ, ψ) ≈ dot(ψ, ψ)  
    @test dot(ψ, ϕ) ≈ conj(dot(ϕ, ψ)) 
   
    @test dot(ψ, Id, ϕ) ≈ conj(dot(ϕ, Id, ψ))
    @test dot(ψ, Id, ϕ) ≈ dot(ψ, ϕ)  

    @test norm(ψ) ≈ sqrt(abs(dot(ψ, ψ)))
end

@testset "left environment" begin
    L = left_env(ϕ, ψ)
    @test tr(L[end]) ≈ dot(ϕ, ψ)  
end

@testset "Cauchy-Schwarz inequality" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end
end