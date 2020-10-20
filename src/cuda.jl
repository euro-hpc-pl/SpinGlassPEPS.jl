
export left_env

function LinearAlgebra.dot(ϕ::MPS{T}, ψ::MPS{T}) where {T <: CuArray}
    C = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @cutensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return C[1]
end

function left_env(ϕ::MPS{T}, ψ::MPS{T}) where {T <: CuArray}
    size = length(ψ)
    S = eltype(ψ)

    L = Vector{CuMatrix{S}}(undef, size+1)
    L[1] = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ 1:size
        M = ψ[i]
        M̃ = conj(ϕ[i])

        C = L[i]
        @cutensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    return L
end

function LinearAlgebra.dot(ϕ::MPS, O::Vector{T}, ψ::MPS) where {T <: CuMatrix}
    S = eltype(ψ)
    C = CUDA.ones(S, 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        Mat = O[i]
        @cutensor C[x, y] := M̃[β, σ, x] * Mat[σ, η] * C[β, α] * M[α, η, y] order = (α, η, β, σ)
    end
    return C[1]
end

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where {T <: CuArray}
    ψ = MPS{T}(L)
    S = eltype(T)
    ψ[1] = CUDA.randn(S, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = CUDA.randn(S, D, d, D)
    end
    ψ[end] = CUDA.randn(S, D, d, 1)
    ψ
end
