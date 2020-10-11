_create_empty(::Type{T}, l) where {T <: CuVector{S}} where {S <: Number} = Vector{CuArray{S, 3}}(undef, l)

function Base.:(*)(ψ′::Adjoint{S, MPS{T}}, ϕ::MPS{T}) where {T <: CuArray{S, 3}} where {S <: Number}
    ψ = ψ′.parent

    M   = ϕ.tensors[1]
    M̃dg = dg(ψ.tensors[1])

    @cutensor cont[b₁, a₁] := M̃dg[b₁, 1, σ₁] * M[1, a₁, σ₁]

    l = length(ϕ)
    for i=2:l-1
        M   = ϕ.tensors[i]
        M̃dg = dg(ψ.tensors[i])

        @cutensor cont[bᵢ, aᵢ] := M̃dg[bᵢ, bᵢ₋₁, σᵢ] * cont[bᵢ₋₁, aᵢ₋₁] * M[aᵢ₋₁, aᵢ, σᵢ]
    end
    M   = ϕ.tensors[l]
    M̃dg = dg(ψ.tensors[l])
    
    @cutensor M̃dg[1, bᴸ⁻¹, σᴸ] * cont[bᴸ⁻¹, aᴸ⁻¹] * M[aᴸ⁻¹, 1, σᴸ]
end

function Base.:(*)(O::MPO, ψ::MPS{T}) where {T <: CuArray}
    tensors = copy(ψ.tensors)
    l = length(O)
    for i in 1:l
        W = O.tensors[i]
        M = ψ.tensors[i]

        @reduce N[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ] :=  sum(σ′ᵢ) W[bᵢ₋₁, bᵢ, σᵢ, σ′ᵢ] * M[aᵢ₋₁, aᵢ, σ′ᵢ]
        
        tensors[i] = N
    end
    MPS(tensors)
end

function Base.:(*)(O1::MPO{T}, O2::MPO{T}) where {T <: CuArray}
    tensors = copy(O1.tensors)
    l = length(O1)
    for i in 1:l
        W1 = O1.tensors[i]
        W2 = O2.tensors[i]

        @reduce V[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ, σ′ᵢ] :=  sum(σ′′ᵢ) W1[bᵢ₋₁, bᵢ, σᵢ, σ′′ᵢ] * W2[aᵢ₋₁, aᵢ, σ′′ᵢ, σ′ᵢ]
        
        tensors[i] = V
    end
    MPO{T}(tensors)
end

function Base.:(*)(ψ′::Adjoint{S, MPS{T}}, O::MPO) where {T <: CuArray{S, 3}} where {S <: Number}
    ψ = ψ′.parent
    tensors = copy(ψ.tensors)
    Ws = dg.(reverse(O.tensors))
    l = length(O)
    for i in 1:l
        W = Ws[i]
        M = ψ.tensors[i]

        @reduce A[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ] :=  sum(σ′ᵢ) W[bᵢ₋₁, bᵢ, σᵢ, σ′ᵢ] * M[aᵢ₋₁, aᵢ, σ′ᵢ]
        tensors[i] = A
    end
    adjoint(MPS{T}(tensors))
end