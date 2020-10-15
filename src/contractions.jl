
export norm 

function LinearAlgebra.dot(ϕ::MPS{T}, ψ::MPS{T}) where T <: AbstractArray{<:Number, 3} 
    S = eltype(ψ)
    C = ones(S, 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return C[1]
end

function LinearAlgebra.norm(ψ::MPS{T}) where T <: AbstractArray{<:Number, 3}
    return sqrt(abs(dot(ψ,ψ)))
end

function LinearAlgebra.dot(ϕ::MPS{T}, O::Vector{S}, ψ::MPS{T}) where {T <: AbstractArray{<:Number, 3}, S <: AbstractArray{<:Number, 2}}
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

function LinearAlgebra.dot(O::MPO{T}, ψ::MPS{S}) where {T <: AbstractArray{<:Number, 4}, S <: AbstractArray{<:Number, 3}}
    L = length(ψ)
    tensors = Vector{T}(undef, L)

    for i in 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), (y, b), σ] := sum(η) W[x, σ, y, η] * M[a, η, b]      
        tensors[i] = N
    end
    MPS(tensors)
end

function Base.:(*)(O::MPO{T}, ψ::MPS{S}) where {T <: AbstractArray{<:Number, 4}, S <: AbstractArray{<:Number, 3}}
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
