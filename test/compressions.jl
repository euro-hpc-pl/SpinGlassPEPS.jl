using TensorOperations

@testset "Canonisation and Compression" begin

D = 20
Dcut = 2

tol = 1E-4
max_sweeps = 5

d = 2
sites = 5

T = Array{ComplexF64, 3}

ψ = randn(MPS{T}, sites, D, d)
ϕ = randn(MPS{T}, sites, D, d)
χ = randn(MPS{T}, sites, D, d)
Φ = randn(MPS{T}, sites, D, d)

@testset "Canonisation (left)" begin
    canonise!(ψ, :left)  
    show(ψ)  
 
    is_left_normalized = true
    for i ∈ 1:length(ψ)
        A = ψ[i]
        D = size(A, 3)

        @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
        is_left_normalized *= I(D) ≈ Id
    end 

    @test is_left_normalized 
    @test dot(ψ, ψ) ≈ 1  
end

@testset "Canonisation (right)" begin
    canonise!(ϕ, :right)  
    show(ϕ)

    is_right_normalized = true
    for i ∈ 1:length(ϕ)
        B = ϕ[i]
        D = size(B, 1)

        @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        is_right_normalized *= I(D) ≈ Id
    end 

    @test is_right_normalized 
    @test dot(ϕ, ϕ) ≈ 1      
end

@testset "Canonisation (both)" begin
    canonise!(χ)  
    show(χ)
    @test dot(χ, χ) ≈ 1      
end

@testset "Truncation (SVD, right)" begin
    truncate!(ψ, :right, Dcut)  
    show(ψ)
    @test dot(ψ, ψ) ≈ 1     
end

@testset "Truncation (SVD, left)" begin
    truncate!(ψ, :left, Dcut)  
    show(ψ)
    @test dot(ψ, ψ) ≈ 1     
end

@testset "Variational compression" begin
    Ψ = compress(Φ, Dcut, tol, max_sweeps)  
    show(Ψ)
    @test dot(Ψ, Ψ) ≈ 1     
end

end