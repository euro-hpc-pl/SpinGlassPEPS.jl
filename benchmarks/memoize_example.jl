using Memoize, LRUCache
using SpinGlassPEPS, TensorOperations, TensorCast
using LinearAlgebra

@memoize function left_env2(ϕ::AbstractMPS, idx::NTuple)
    l = length(idx)
    if l == 0
        L = [1.]
    else
        m = idx[l]
        new_idx = idx[1:l-1]
        L_old = left_env2(ϕ, new_idx)
        M = ϕ[l]
        @reduce L[x] := sum(α) L_old[α] * M[α, $m, x]
    end
    return L
end

@memoize function right_env2(ϕ::AbstractMPS, W::AbstractMPO, idx::NTuple)
    l = length(idx)
    L = length(ϕ)
    if l == 0
        R = fill(1., 1, 1)
    else
        m = idx[1]
        new_idx = idx[2:l]
        R_old = right_env2(ϕ, W, new_idx)
        M = ϕ[L-l+1]
        M̃ = W[L-l+1]
        @reduce R[x, y] := sum(α, β, γ) M̃[y, $m, β, γ] * M[x, γ, α] * R_old[α, β]
    end
    return R
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

ϕ = randn(MPS{Float64}, 10, 10, 10);
ψ = randn(MPS{Float64}, 10, 10, 10);
W = randn(MPO{Float64}, 10, 10, 10);
