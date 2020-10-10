@testset "MPS" begin

@testset "creation from vector" begin
    d = 10
    a = randn(ComplexF64, d)
    sites = 5
    ψ = MPS(a, sites)

    @test length(ψ) == sites
    @test size(ψ) == (sites, )
    @test ψ == ψ
    @test ψ ≈ ψ
    @test eltype(ψ) == ComplexF64
    @test ψ[1][1, 1, :] == a
    @test 2 * ψ == MPS(2a, sites)
    @test ψ / 2 == MPS(a/2, sites)
    @test copy(ψ) == ψ
    @show ψ
end

@testset "creation from vector on cuda" begin
    d = 10
    a = CUDA.randn(Float32, d)
    sites = 5
    ψ = MPS(a, sites)

    @test length(ψ) == sites
    @test ψ == ψ
    @test ψ ≈ ψ
    @test eltype(ψ) == Float32
    @test typeof(ψ.tensors[1]) <: CuArray
    @test 2 * ψ == MPS(2a, sites)
    @test ψ / 2 == MPS(a/2, sites)
    @test copy(ψ) == ψ
end

@testset "adjoints" begin
    d = 10
    a = randn(ComplexF32, d)
    sites = 5
    ψ = MPS(a, sites)
    ϕ = ψ'
    @test real(ϕ*ψ) > 0
    @show ϕ

    a = cu(a)
    ψ = MPS(a, sites)
    ϕ = ψ'
    @test real(ϕ*ψ) > 0
end

end