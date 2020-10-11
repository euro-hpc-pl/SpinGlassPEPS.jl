@testset "cuda" begin
    d = 10
    a = CUDA.randn(Float32, d)
    sites = 5
    @testset "creation from vector on cuda" begin
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
        ψ = MPS(a, sites)
        ϕ = ψ'
        @test real.(ϕ*ψ) > 0
    end

end