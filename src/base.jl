export bond_dimension, is_left_normalized, is_right_normalized
export verify_bonds, verify_physical_dims, tensor, rank, physical_dim
export State, dropindices

const State = Union{Vector, NTuple}

abstract type AbstractTensorNetwork{T} end

for (T, N) ∈ ((:PEPSRow, 5), (:MPO, 4), (:MPS, 3))
    AT = Symbol(:Abstract, T)
    @eval begin
        export $AT
        export $T

        abstract type $AT{T} <: AbstractTensorNetwork{T} end

        struct $T{T <: Number} <: $AT{T}
            tensors::Vector{Array{T, $N}}
        end

        # consturctors
        $T(::Type{T}, L::Int) where {T} = $T(Vector{Array{T, $N}}(undef, L))
        $T(L::Int) = $T(Float64, L)

        @inline Base.setindex!(a::$AT, A::AbstractArray{<:Number, $N}, i::Int) = a.tensors[i] = A
        @inline bond_dimension(a::$AT) = maximum(size.(a.tensors, $N))
        Base.copy(a::$T) = $T(copy(a.tensors))

        @inline Base.eltype(::$AT{T}) where {T} = T
    end
end

@inline Base.:(==)(a::AbstractTensorNetwork, b::AbstractTensorNetwork) = a.tensors == b.tensors
@inline Base.:(≈)(a::AbstractTensorNetwork, b::AbstractTensorNetwork)  = a.tensors ≈ b.tensors

@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)
@inline Base.iterate(a::AbstractTensorNetwork) = iterate(a.tensors)
@inline Base.iterate(a::AbstractTensorNetwork, state) = iterate(a.tensors, state)
@inline Base.lastindex(a::AbstractTensorNetwork) = lastindex(a.tensors)
@inline Base.length(a::AbstractTensorNetwork) = length(a.tensors)
@inline Base.size(a::AbstractTensorNetwork) = (length(a.tensors), )
@inline Base.eachindex(a::AbstractTensorNetwork) = eachindex(a.tensors)

@inline LinearAlgebra.rank(ψ::AbstractMPS) = Tuple(size(A, 2) for A ∈ ψ)
@inline physical_dim(ψ::AbstractMPS, i::Int) = size(ψ[i], 2)

@inline MPS(A::AbstractArray) = MPS(A, :right)
@inline MPS(A::AbstractArray, s::Symbol, args...) = MPS(A, Val(s), typemax(Int), args...)
@inline MPS(A::AbstractArray, s::Symbol, Dcut::Int, args...) = MPS(A, Val(s), Dcut, args...)
@inline MPS(A::AbstractArray, ::Val{:right}, Dcut::Int, args...) = _left_sweep_SVD(MPS, A, Dcut, args...)
@inline MPS(A::AbstractArray, ::Val{:left}, Dcut::Int, args...) = _right_sweep_SVD(MPS, A, Dcut, args...)

@inline dropindices(ψ::AbstractMPS, i::Int=2) = (dropdims(A, dims=i) for A ∈ ψ)

function MPS(states::Vector{Vector{T}}) where {T <: Number}
    state_arrays = [reshape(copy(v), (1, length(v), 1)) for v ∈ states]
    MPS(state_arrays)
end

function (::Type{T})(ψ::AbstractMPS) where {T <:AbstractMPO}
    _verify_square(ψ)
    T([
        @cast W[x, σ, y, η] |= A[x, (σ, η), y] (σ:isqrt(size(A, 2)))
        for A in ψ
    ])
end

function (::Type{T})(O::AbstractMPO) where {T <:AbstractMPS}
    T([@cast A[x, (σ, η), y] := W[x, σ, y, η] for W in O])
end

function Base.randn(::Type{MPS{T}}, D::Int, rank::Union{Vector, NTuple}) where {T}
    MPS([
        randn(T, 1, first(rank), D),
        randn.(T, D, rank[begin+1:end-1], D)...,
        rand(T, D, last(rank), 1)
    ])
end

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where {T}
    MPS([
        randn(T, 1, d, D), (randn(T, D, d, D) for _ in 2:L-1)..., randn(T, D, d, 1)
    ])
end

Base.randn(::Type{MPS}, args...) = randn(MPS{Float64}, args...)

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where {T}
    MPO(randn(MPS{T}, L, D, d^2))
end

Base.randn(::Type{MPO}, args...) = randn(MPO{Float64}, args...)

is_left_normalized(ψ::MPS) = all(
    I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ) for A ∈ ψ
)

is_right_normalized(ϕ::MPS) = all(
    I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ) for B in ϕ
)

function _verify_square(ψ::AbstractMPS)
    dims = physical_dim.(Ref(ψ), eachindex(ψ))
    @assert isqrt.(dims) .^ 2 == dims "Incorrect MPS dimensions"
end

function verify_physical_dims(ψ::AbstractMPS, dims::NTuple)
    for i ∈ eachindex(ψ)
        @assert physical_dim(ψ, i) == dims[i] "Incorrect physical dim at site $(i)."
    end
end

function verify_bonds(ψ::AbstractMPS)
    L = length(ψ)

    @assert size(ψ[1], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ[end], 3) == 1 "Incorrect size on the right boundary."

    for i ∈ 1:L-1
        @assert size(ψ[i], 3) == size(ψ[i+1], 1) "Incorrect link between $i and $(i+1)."
    end
end

function Base.show(io::IO, ψ::AbstractTensorNetwork)
    L = length(ψ)
    dims = [size(A) for A ∈ ψ]

    println(io, "Matrix product state on $L sites:")
    _show_sizes(io, dims)
    println(io, "   ")
end


function _show_sizes(io::IO, dims::Vector, sep::String=" x ", Lcut::Int=8)
    L = length(dims)
    if L > Lcut
        for i ∈ 1:Lcut
            print(io, " ", dims[i], sep)
        end
        print(io, " ... × ", dims[end])
    else
        for i ∈ 1:(L-1)
            print(io, dims[i], sep)
        end
        println(io, dims[end])
    end
end
