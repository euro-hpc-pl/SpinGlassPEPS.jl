using Memoize, LRUCache
using SpinGlassPEPS, TensorOperations
using LinearAlgebra

function left_env2(ϕ::AbstractMPS, ψ::AbstractMPS) 
    l = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))

    L = Vector{Matrix{T}}(undef, l+1)
    L[1] = ones(eltype(ψ), 1, 1)

    for i ∈ 1:l
        L[i+1] = left_env2(ϕ, ψ, i, L[i])
    end
    L
end

@memoize function left_env2(ϕ::AbstractMPS, ψ::AbstractMPS, i::Int, L::AbstractArray)
    M = ψ[i]
    M̃ = conj.(ϕ[i])
    @tensor C[x, y] := M̃[β, σ, x] * L[β, α] * M[α, σ, y] order = (α, β, σ)
    C
end

@memoize function left_env3(ϕ::AbstractMPS, ψ::AbstractMPS) 
    l = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))

    L = Vector{Matrix{T}}(undef, l+1)
    L[1] = ones(eltype(ψ), 1, 1)

    for i ∈ 1:l
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        C = L[i]
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    L
end

ϕ = randn(MPS{Float64}, 100, 100, 100);
ψ = randn(MPS{Float64}, 100, 100, 100);

left_env(ϕ, ψ);
left_env2(ϕ, ψ);
left_env3(ϕ, ψ);