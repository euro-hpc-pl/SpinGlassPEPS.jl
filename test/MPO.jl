@testset "MPO" begin

D = 9
d = 2
sites = 5
T = Array{ComplexF64, 4}

@testset "Random MPO" begin
    O = randn(MPO{T}, sites, D, d)

    @test O == O
    @test O ≈ O

    @test length(O) == sites
    @test size(O) == (sites, )
    @test eltype(O) == ComplexF64

    #@test P = copy(O)
    #@test P == O 
    #@test P ≈ O 
end

end