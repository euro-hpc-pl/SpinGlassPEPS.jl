@testset "contractions" begin

d = 10
a = randn(ComplexF64, d)
b = randn(ComplexF64, d)
sites = 5

@testset "dot products" begin
    ψ = MPS(a, sites)
    ϕ = MPS(b, sites)

    @test ϕ'*ψ == dot(ϕ, ψ) 
end

end