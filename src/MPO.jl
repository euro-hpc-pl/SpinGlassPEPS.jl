export MPO

struct MPO{T <: AbstractArray{<:Number, 4}}
    tensors::Vector{T}
end

# consturctors

newDim(::Type{T}, dims) where {T<:AbstractArray} = T.name.wrapper{eltype(T), dims}

function MPO(ψ::MPS{T}) where T <: AbstractArray{<:Number, 3}
    L = length(ψ)

    S = newDim(T, 4) 
    tensors = Vector{S}(undef, L)

    for i ∈ 1:L
        A = ψ[i]
        dd = size(A, 2)
        d = isqrt(dd)
        
        if d^2 == dd
            @cast W[x, σ, y, η] |= A[x, (σ, η), y] (σ:d) 
            tensors[i] = W
        else
            error("$dd is not a square number.")
        end        
    end 
    MPO(tensors) 
end 

function MPS(O::MPO{T}) where T <: AbstractArray{<:Number, 4}
    L = length(O)

    S = newDim(T, 3) 
    tensors = Vector{S}(undef, L)

    for i ∈ 1:L
        W = O[i]
        @cast A[x, (σ, η), y] := W[x, σ, y, η]
        tensors[i] = A     
    end 
    MPS(tensors) 
end  

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where T <: AbstractArray{<:Number, 4}
    S = newDim(T, 3) 
    ψ = randn(MPS{S}, L, D, d^2) 
    MPO(ψ)
end

# length, comparison, element types

Base.:(==)(O::MPO, U::MPO) = O.tensors == U.tensors
Base.:(≈)(O::MPO, U::MPO)  = O.tensors ≈ U.tensors

Base.eltype(::Type{MPO{T}}) where {T <: AbstractArray{S, 4}} where {S <: Number} = S

Base.getindex(O::MPO) = getindex(O.tensors, )
Base.setindex(O::MPO, i::Int) = setindex(O.tensors, i)

Base.length(O::MPO) = length(O.tensors)
Base.size(O::MPO) = (length(O.tensors), )

Base.copy(O::MPO) = MPO(O.tensors)

# printing