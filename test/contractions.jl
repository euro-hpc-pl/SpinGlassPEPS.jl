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
end

@testset "left environment" begin
    L = left_env(ϕ, ψ)
    @test L[end][1] ≈ dot(ϕ, ψ)  
end

@testset "Cauchy-Schwarz inequality" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
end

end