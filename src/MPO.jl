export MPO

struct MPO{T <: AbstractArray{<:Number, 4}}
    tensors::Vector{T}

    function MPO{T}(L::Int) where {T}
        new(Vector{T}(undef, L))
    end

    MPO{T}(v::Vector{T}) where {T} = new(v)
end

# consturctors

MPO(::Type{T}, ::Type{S}, L::Int) where {T<:AbstractArray, S<:Number} = MPO{T{S, 4}}(L)

function MPO(ψ::MPS{T}) where {T <: AbstractArray{<:Number, 3}}
    _verify_square(ψ)
    L = length(ψ)
    O = MPO(T.name.wrapper, eltype(T), L)

    for i ∈ 1:L
        A = ψ[i]
        d = isqrt(size(A, 2))
        
        @cast W[x, σ, y, η] |= A[x, (σ, η), y] (σ:d) 
        O[i] = W
    end 
    O
end 

function MPS(O::MPO{T}) where T <: AbstractArray{<:Number, 4}
    L = length(O)
    ψ = MPS(T.name.wrapper, eltype(T), L)

    for i ∈ 1:L
        W = O[i]
        @cast A[x, (σ, η), y] := W[x, σ, y, η]
        ψ[i] = A     
    end 
    ψ
end  

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where T <: AbstractArray{<:Number, 4}
    S = newdim(T, 3)
    ψ = randn(MPS{S}, L, D, d^2) 
    MPO(ψ)
end

function _verify_square(ψ::MPS)
    arr = [size(A, 2) for A ∈ ψ]
    @assert isqrt.(arr) .^ 2 == arr "Incorrect MPS dimensions"
end


# length, comparison, element types

Base.:(==)(O::MPO, U::MPO) = O.tensors == U.tensors
Base.:(≈)(O::MPO, U::MPO)  = O.tensors ≈ U.tensors

Base.eltype(::Type{MPO{T}}) where {T} = eltype(T)

Base.getindex(O::MPO, i::Int) = getindex(O.tensors, i)
Base.setindex!(O::MPO, W::AbstractArray{<:Number, 4}, i::Int) = O.tensors[i] = W

Base.iterate(O::MPO) = iterate(O.tensors)
Base.iterate(O::MPO, state) = iterate(O.tensors, state)
Base.lastindex(O::MPO) = lastindex(O.tensors)

Base.length(O::MPO) = length(O.tensors)
Base.size(O::MPO) = (length(O.tensors), )

Base.copy(O::MPO) = MPO(O.tensors)

# printing