@testset "MPS" begin

d = 10
a = randn(ComplexF64, d)
sites = 5

@testset "creation from vector" begin
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

@testset "adjoints" begin
    ψ = MPS(a, sites)
    ϕ = ψ'
    @test real(ϕ*ψ) > 0
    @show ϕ
end

end