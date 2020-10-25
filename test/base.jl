@testset "MPS" begin

D = 10
d = 4
sites = 5
T = ComplexF64

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

@testset "Random MPO" begin
    O = randn(MPO{T}, sites, D, d)

    @test O == O
    @test O ≈ O

    @test length(O) == sites
    @test size(O) == (sites, )
    @test eltype(O) == ComplexF64

    P = copy(O)
    @test P == O 
    @test P ≈ O 
end

end