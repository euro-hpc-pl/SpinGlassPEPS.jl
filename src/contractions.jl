
export dot, norm

function LinearAlgebra.dot(ϕ::MPS, ψ::MPS)
    C = ones(eltype(ψ), 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return C[1]
end

LinearAlgebra.norm(ψ::MPS) = sqrt(abs(dot(ψ, ψ)))

<<<<<<< HEAD
function LinearAlgebra.dot(ϕ::MPS{T}, O::Vector{S}, ψ::MPS{T}) where {T <: AbstractArray{<:Number, 3}, S <: AbstractArray{<:Number, 2}}
=======
function LinearAlgebra.dot(ϕ::MPS, O::Vector{S}, ψ::MPS) where {S <: AbstractMatrix}
>>>>>>> 19fa41ae9126f12b46ec0d2b7b43843e5c3732f4
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

function LinearAlgebra.dot(O::MPO, ψ::MPS{T}) where {T}
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

function LinearAlgebra.dot(O1::MPO{T}, O2::MPO{T}) where T <: AbstractArray{<:Number, 4}
    L = length(O1)
    tensors = Vector{T}(undef, L)

    for i in 1:L
        W1 = O1.tensors[i]
        W2 = O2.tensors[i]

        @reduce V[(x, a), (y, b), σ, η] := sum(γ) W1[x, y, σ, γ] * W2[a, b, γ, η]        
        tensors[i] = V
    end
    MPO(tensors)
end

function Base.:(*)(O1::MPO{T}, O2::MPO{T}) where T <: AbstractArray{<:Number, 4}
    return dot(O1, O2)
end
