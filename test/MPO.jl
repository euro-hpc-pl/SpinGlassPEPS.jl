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

@testset "contractions" begin
    g = 1.0; L = 7

    id = [1  0; 0  1]
    σˣ = [0  1;  1  0]
    σᶻ = [1  0;  0 -1]
    W_tnsr = zeros(ComplexF64, 3, 3, 2, 2)
    W_tnsr[1, 1, :, :] = id    
    W_tnsr[2, 1, :, :] = -σᶻ  
    W_tnsr[3, 1, :, :] = -g*σˣ
    W_tnsr[3, 2, :, :] = σᶻ   
    W_tnsr[3, 3, :, :] = id 
    H = MPO(W_tnsr, L)

    ψ = MPS(W_tnsr[1, 1 ,1, :], L)
    @test typeof(H*H) <: MPO
    @test typeof(H*ψ) <: MPS
    @test typeof(ψ' * H) <: Adjoint
    
    
end

@testset "printing" begin
    H = MPO(a, sites)
    @show H
    println(H)
end
end