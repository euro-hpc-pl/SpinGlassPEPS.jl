@testset "multiplication of identities" begin
    ψ = randn(MPS{Float64}, 4, 3, 2)
    O = randn(MPO{Float64}, 4, 3, 2)

    IMPS = MPS(I)
    IMPO = MPO(I)

    @test IMPO * ψ == ψ
    @test IMPO * O == O
    
    ϕ = O * IMPS

    @test typeof(ϕ) == MPS{Float64}
    @test length(ϕ) == length(O)

    for i ∈ eachindex(O)
        @test ϕ[i] == dropdims(sum(O[i], dims=4), dims=4)
    end

    @test IMPO === MPO(I)
end