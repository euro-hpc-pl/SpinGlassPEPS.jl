@testset "MPS" begin

D = 10
d = 4
sites = 5
T = Array{ComplexF64, 3}

@testset "Random MPS" begin
    ψ = randn(MPS{T}, sites, D, d)

    @test ψ == ψ
    @test ψ ≈ ψ

    @test length(ψ) == sites
    @test size(ψ) == (sites, )
    @test eltype(ψ) == ComplexF64

    ϕ = copy(ψ) 
    @test ϕ == ψ
    @test ϕ ≈ ψ

    show(ψ)
end
end