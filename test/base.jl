@testset "MPS" begin

D = 10
d = 4
sites = 5
T = ComplexF64

@testset "Random MPS" begin
    ψ = randn(MPS{T}, sites, D, d)
    @test _verify_bonds(ψ)

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

@testset "Retrieving bond dimension" begin
    S = Float64
    size = 5
    D = 6
    d = 2
    ψ = randn(MPS{S}, sites, D, d)

    @test bond_dimension(ψ) ≈ D
end

@testset "MPS from Product state" begin
    L = 3

    prod_state = [ [1, 1.] / sqrt(2) for _ ∈ 1:L]
    ϕ = MPS(prod_state)

    show(ϕ)
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