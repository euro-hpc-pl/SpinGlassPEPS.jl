using TensorOperations
include("../src/compression.jl")
@testset "Canonisation and Compression" begin

D = 10
Dcut = 5

d = 2
sites = 5

T = Float64

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
        DD = size(A, 3)

        @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
        is_left_normalized *= I(DD) ≈ Id
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
        DD = size(B, 1)

        @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        is_right_normalized *= I(DD) ≈ Id
    end 

    @test is_right_normalized 
    @test dot(ϕ, ϕ) ≈ 1      
end

@testset "Cauchy-Schwarz inequality (after truncation)" begin
    @test abs(dot(ϕ, ψ)) <= norm(ϕ) * norm(ψ)
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
    Dcut = 5
    tol = 1E-4
    max_sweeps = 5

    canonise!(Φ, :right)
    @test dot(Φ, Φ) ≈ 1 

    Ψ = compress(Φ, Dcut, tol, max_sweeps)  

    show(Ψ)
    @test dot(Ψ, Ψ) ≈ 1    
    
    println("(Ψ, Ψ) = ", dot(Ψ, Ψ))
    println("(Φ, Φ) = ", dot(Φ, Φ))

    overlap = dot(Ψ, Φ)
    dist1 = 2 - 2 * real(overlap)
    dist2 = norm(Ψ)^2 + norm(Φ)^2 - 2 * real(overlap)

    @test abs(dist1 - dist2) < 1e-14

    println("(Φ, Ψ) = ", overlap)
    println("dist(Φ, Ψ)^2 = ", dist2)
end

@testset "Compare with Krzysiek's implementation" begin
    Dcut = 2
    tol = 1E-4
    max_sweeps = 5

    canonise!(Φ, :right)
    @test dot(Φ, Φ) ≈ 1 

    Ψ = compress(Φ, Dcut, tol, max_sweeps)  

    Φ_trunc = copy(Φ)
    truncate!(Φ_trunc, :right, Dcut)

    permuted_mps = map(x->permutedims(x, (1,3,2)), Φ.tensors)
    # tensors = compress_mps_itterativelly(, Φ_trunc.tensors, Dcut, tol)
    tensors = compress_iter(permuted_mps, Dcut, tol)
    tensors = map(x->permutedims(x, (1,3,2)), tensors)
    ξ = MPS(tensors)
    # ξ.tensors = tensors
     
    @test dot(ξ, ξ) ≈ 1   

    println("(ξ, ξ) = ", dot(ξ, ξ))

    overlap = dot(Ψ, ξ)
    dist1 = 2 - 2 * real(overlap)
    dist2 = norm(Ψ)^2 + norm(ξ)^2 - 2 * real(overlap)

    @test abs(dist1 - dist2) < 1e-14
        
    println("Krzysiek wins - flawless victory, fatality.")
end
end