@testset "MPO" begin

d = 10
a = randn(ComplexF64, d, d, d, d)
sites = 5

@testset "creation" begin
    H = MPO(a, sites)
    @test H == H
    @test H ≈ H
    @test H[1] == H.tensors[1]
    @test length(H) == sites
end

@testset "printing" begin
    H = MPO(a, sites)
    @show H
    println(H)
end
end