@testset "cuda" begin
    d = 10
    a = CUDA.randn(Float32, d)
    sites = 5
    @testset "creation of MPS from vector on cuda" begin
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

    @testset "adjoints of MPS" begin
        ψ = MPS(a, sites)
        ϕ = ψ'
        @test real.(ϕ*ψ) > 0
    end

    @testset "MPO on cuda" begin
        d = 10
        a = CUDA.rand(Float32, d, d, d, d)
        sites = 5
        H = MPO(a, sites)
        @test typeof(H.tensors[1]) <: CuArray
    end

end