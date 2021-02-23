ψ = randn(MPS{Float64}, 4, 3, 2)
O = randn(MPO{Float64}, 4, 3, 2)

IMPS = MPS(I)
IMPO = MPO(I)

@testset "multiplication of IdentityMPO" begin

    @testset "mutlitplication with MPS ψ returns ψ" begin
        @test IMPO * ψ == ψ
        @test ψ * IMPO == ψ
    end

    @testset "mutlitplication with MPO O returns O" begin
        @test IMPO * O == O
    end
end

@testset "Multiplication of IdentityMPS by an MPO O" begin
    ϕ = O * IMPS

    @testset "result has the correct type" begin
        @test typeof(ϕ) == MPS{Float64}
    end

    @testset "length of result is the same as O" begin
        @test length(ϕ) == length(O)
    end

    @testset "the multiplication drops the correct dims" begin
        for i ∈ eachindex(O)
            @test ϕ[i] == dropdims(sum(O[i], dims=4), dims=4)
        end
    end
end

@testset "Identities are singletons" begin
    @test IMPO === MPO(I)
    @test IMPS === MPS(I)
end

@testset "Identities have infinite length" begin
    @test length(IMPS) == Inf
    @test length(IMPO) == Inf
end

@testset "Indexing identities returns trivial tensors" begin
    @testset "Indexing IdentityMPS" begin
        A = IMPS[42]
        @test length(A) == 1
        @test ndims(A) == 3
        @test norm(A) == 1
    end

    @testset "Indexing IdentityMPO" begin
        B = IMPO[666]
        @test length(B) == 1
        @test ndims(B) == 4
        @test norm(B) == 1
    end
end