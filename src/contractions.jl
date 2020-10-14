
export norm 

function LinearAlgebra.dot(ϕ::MPS, ψ::MPS)
    C = ones(eltype(ψ), 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return C[1]
end

function LinearAlgebra.norm(ψ::MPS)
    return sqrt(abs(dot(ψ, ψ)))
end

function LinearAlgebra.dot(ϕ::MPS, O::Vector{S}, ψ::MPS) where {S <: AbstractMatrix}
    R = eltype(ψ)
    C = ones(R, 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        Mat = O[i]
        @tensor C[x, y] := M̃[β, σ, x] * Mat[σ, η] * C[β, α] * M[α, η, y] order = (α, η, β, σ)
    end
    return C[1]
end

function dot(O::MPO, ψ::MPS{T}) where {T}
    L = length(ψ)
    ϕ = MPS{T}(L)

    for i in 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), (y, b), σ] := sum(η) W[x, σ, y, η] * M[a, η, b]      
        ϕ[i] = N
    end
    ϕ
end

function Base.:(*)(O::MPO, ψ::MPS)
    return dot(O, ψ)
end