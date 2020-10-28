using Base
export bond_dimension
export _verify_bonds

for (T, N) in ((:MPO, 4), (:MPS, 3))
    AT = Symbol(:Abstract, T)
    @eval begin
        export $AT
        export $T

        abstract type $AT{T} end

        struct $T{T <: Number} <: $AT{T}
            tensors::Vector{Array{T, $N}}
        end
        
        # consturctors
        $T(::Type{T}, L::Int) where {T} = $T(Vector{Array{T, $N}}(undef, L))
        $T(L::Int) = $T(Float64, L)

        Base.setindex!(a::$AT, A::AbstractArray{<:Number, $N}, i::Int) = a.tensors[i] = A
        bond_dimension(a::$AT) = maximum(size.(a.tensors, $N))
        Base.copy(a::$T) = $T(copy(a.tensors))
        Base.eltype(::$AT{T}) where {T} = T
    end
end

const AbstractMPSorMPO = Union{AbstractMPS, AbstractMPO}
const MPSorMPO = Union{MPS, MPO}

Base.:(==)(a::AbstractMPSorMPO, b::AbstractMPSorMPO) = a.tensors == b.tensors
Base.:(≈)(a::AbstractMPSorMPO, b::AbstractMPSorMPO)  = a.tensors ≈ b.tensors

Base.getindex(a::AbstractMPSorMPO, i::Int) = getindex(a.tensors, i)
Base.iterate(a::AbstractMPSorMPO) = iterate(a.tensors)
Base.iterate(a::AbstractMPSorMPO, state) = iterate(a.tensors, state)
Base.lastindex(a::AbstractMPSorMPO) = lastindex(a.tensors)
Base.length(a::AbstractMPSorMPO) = length(a.tensors)
Base.size(a::AbstractMPSorMPO) = (length(a.tensors), )


function MPS(vec::Vector{Vector{T}}) where  {T <: Number}
    L = length(vec)
    ψ = MPS(T, L)
    for i ∈ 1:L
           A = reshape(vec[i], 1, :, 1)
        ψ[i] = copy(A)
    end    
    return ψ
end

function MPO(ψ::MPS)
    _verify_square(ψ)
    L = length(ψ)
    O = MPO(eltype(ψ), L)

    for i ∈ 1:L
        A = ψ[i]
        d = isqrt(size(A, 2))
        
        @cast W[x, σ, y, η] |= A[x, (σ, η), y] (σ:d) 
        O[i] = W
    end 
    O
end 

function MPS(O::MPO)
    L = length(O)
    ψ = MPS(eltype(O), L)

    for i ∈ 1:L
        W = O[i]
        @cast A[x, (σ, η), y] := W[x, σ, y, η]
        ψ[i] = A     
    end 
    return ψ
end  

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = MPS(T, L)
    ψ[1] = randn(T, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = randn(T, D, d, D)
    end
    ψ[end] = randn(T, D, d, 1)
    return ψ
end

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = randn(MPS{T}, L, D, d^2) 
    MPO(ψ)
end

function _verify_square(ψ::AbstractMPS)
    arr = [size(A, 2) for A ∈ ψ]
    @assert isqrt.(arr) .^ 2 == arr "Incorrect MPS dimensions"
end

function _verify_bonds(ψ::AbstractMPS)
    L = length(ψ)

    @assert size(ψ[1], 1) == 1 "Incorrect size on the left boundary." 
    @assert size(ψ[end], 3) == 1 "Incorrect size on the right boundary." 

    for i ∈ 1:L-1
        @assert size(ψ[i], 3) == size(ψ[i+1], 1) "Incorrect link between $i and $(i+1)." 
    end     
end     

function Base.show(::IO, ψ::AbstractMPS)
    L = length(ψ)
    σ_list = [size(ψ[i], 2) for i ∈ 1:L] 
    χ_list = [size(ψ[i][:, 1, :]) for i ∈ 1:L]
 
    println("Matrix product state on $L sites:")
    println("Physical dimensions: ")
    _show_sizes(σ_list)
    println("   ")
    println("Bond dimensions:   ")
    _show_sizes(χ_list)
end

function _show_sizes(dims::Vector, sep::String=" x ", Lcut::Int=8)
    L = length(dims)
    if L > Lcut
        for i ∈ 1:Lcut
            print(" ", dims[i], sep)
        end
        print(" ... × ", dims[end])
    else
        for i ∈ 1:(L-1)
            print(dims[i], sep)
        end
        println(dims[end])
    end
end   
