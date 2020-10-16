
export dot, norm, left_env

function LinearAlgebra.dot(ϕ::MPS, ψ::MPS)
    C = ones(eltype(ψ), 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return C[1]
end

function left_env(ϕ::MPS, ψ::MPS) 
    size = length(ψ)
    T = eltype(ψ)

    L = Vector{AbstractMatrix{T}}(undef, size+1)
    L[1] = ones(eltype(ψ), 1, 1)

    for i ∈ 1:size
        M = ψ[i]
        M̃ = conj(ϕ[i])

        C = L[i]
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    return L
end

LinearAlgebra.norm(ψ::MPS) = sqrt(abs(dot(ψ, ψ)))

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

Base.:(*)(O::MPO, ψ::MPS) = return dot(O, ψ)

function LinearAlgebra.dot(O1::MPO{T}, O2::MPO{T}) where T <: AbstractArray{<:Number, 4}
    L = length(O1)
    tensors = Vector{T}(undef, L)

    for i in 1:L
        W1 = O1[i]
        W2 = O2[i]
        
        @reduce V[(x, a), (y, b), σ, η] := sum(γ) W1[x, y, σ, γ] * W2[a, b, γ, η]        
        O[i] = V
    end
    MPO(tensors)
end

Base.:(*)(O1::MPO, O2::MPO) = dot(O1, O2)
