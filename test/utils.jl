@testset "HadamardMPS" begin
    rank = (2, 3, 4)
    T = Float64
    ψ = HadamardMPS(T, rank)
    
    @testset "Has correct size" begin
        @test size(ψ) == (length(rank), )
    end
end
