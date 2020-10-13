@testset "contractions" begin

    d = 2
    a = randn(ComplexF64, d)
    sites = 5

    @testset "compression" begin
        ψ = MPS(a, sites)
        @test_nowarn ϕ = _right_compress(ψ)     
        @test_nowarn ϕ = _left_compress(ψ) 
    end
end