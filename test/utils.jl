@testset "HadamardMPS" begin
    L = 10
    ψ = HadamardMPS(L)
    
    @testset "Has correct length" begin
        @test length(ψ) == L
    end

    @testset "Is normalized" begin
        @test norm(ψ) ≈ 1.
    end
end
