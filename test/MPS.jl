@testset "MPS" begin

@testset "creation from vector" begin
    d = 10
    a = randn(ComplexF64, d)
    sites = 5
    ψ = MPS(a, sites)

    @test length(ψ) == sites
    @test ψ == ψ
    @test ψ ≈ ψ
    @test eltype(ψ) == ComplexF64
    @test ψ[1][1, 1, :] == a
    @test 2 * ψ == MPS(2a, sites)
    @test ψ / 2 == MPS(a/2, sites)
    @test copy(ψ) == ψ
end

# @testset "adjoints" begin
    
# end

end