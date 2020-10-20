@testset "cuda" begin
    sites = 10
    @testset "creation of random MPS on cuda" begin
        ψ = randn(MPS{CuArray{Float32, 3}}, sites, 10, 2)

        @test length(ψ) == sites
        @test ψ == ψ
        @test ψ ≈ ψ
        @test eltype(ψ) == Float32
        @test typeof(ψ.tensors[1]) <: CuArray
        @test copy(ψ) == ψ
    end


    @testset "random MPO on cuda" begin
        H = randn(MPO{CuArray{Float32, 4}}, sites, 10, 2)
        @test typeof(H.tensors[1]) <: CuArray
    end

    @testset "contraction on cuda" begin
        ψ = randn(MPS{CuArray{Float32, 3}}, sites, 10, 2)
        H = randn(MPO{CuArray{Float32, 4}}, sites, 10, 2)

        @test typeof((H*H).tensors[1]) <: CuArray
        @test typeof((H*ψ).tensors[1]) <: CuArray
        @test typeof((ψ' * H).parent.tensors[1]) <: CuArray
    end
end