export bond_dimension, is_left_normalized, is_right_normalized
export verify_bonds, verify_physical_dims, tensor, rank

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

LinearAlgebra.rank(ψ::MPS) = Tuple(size(A, 2) for A ∈ ψ)

MPS(A::AbstractArray) = MPS(A, :right)
MPS(A::AbstractArray, s::Symbol, args...) = MPS(A, Val(s), typemax(Int), args...)
MPS(A::AbstractArray, s::Symbol, Dcut::Int, args...) = MPS(A, Val(s), Dcut, args...)
MPS(A::AbstractArray, ::Val{:right}, Dcut::Int, args...) = _left_sweep_SVD(A, Dcut, args...)
MPS(A::AbstractArray, ::Val{:left}, Dcut::Int, args...) = _right_sweep_SVD(A, Dcut, args...)

function _right_sweep_SVD(Θ::AbstractArray{T}, Dcut::Int=typemax(Int), args...) where {T}
    rank = ndims(Θ)
    ψ = MPS(T, rank)

    V = reshape(copy(conj(Θ)), (length(Θ), 1))

    for i ∈ 1:rank
        d = size(Θ, i)

        # reshape
        @cast M[(x, σ), y] |= V'[x, (σ, y)] (σ:d)
       
        # decompose
        U, Σ, V = svd(M, Dcut, args...)
        V *= Diagonal(Σ)

        # create MPS  
        @cast A[x, σ, y] |= U[(x, σ), y] (σ:d)
        ψ[i] = A
    end
    ψ
end

function _left_sweep_SVD(Θ::AbstractArray{T}, Dcut::Int=typemax(Int), args...) where {T}
    rank = ndims(Θ)
    ψ = MPS(T, rank)

    U = reshape(copy(Θ), (length(Θ), 1))

    for i ∈ rank:-1:1
        d = size(Θ, i)

        # reshape
        @cast M[x, (σ, y)] |= U[(x, σ), y] (σ:d)

        # decompose
        U, Σ, V = svd(M, Dcut, args...)
        U *= Diagonal(Σ)

        # create MPS  
        @cast B[x, σ, y] |= V'[x, (σ, y)] (σ:d)
        ψ[i] = B
    end
    ψ
end 

function tensor(ψ::MPS, state::Union{Vector, NTuple})
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end 
    Θ    
end

function MPS(states::Vector{Vector{T}}) where {T <: Number}
    L = length(states)
    ψ = MPS(T, L)
    for i ∈ 1:L
        v = states[i]
        ψ[i] = reshape(copy(v), (1, length(v), 1))
    end
    ψ
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
    ψ
end  

function Base.randn(::Type{MPS{T}}, D::Int, rank::Union{Vector, NTuple}) where {T}
    L = length(rank)
    ψ = MPS(T, L)
    ψ[1] = randn(T, 1, rank[1], D)
    for i ∈ 2:(L-1)
        ψ[i] = randn(T, D, rank[i], D)
    end
    ψ[end] = randn(T, D, rank[end], 1)
    ψ
end

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = MPS(T, L)
    ψ[1] = randn(T, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = randn(T, D, d, D)
    end
    ψ[end] = randn(T, D, d, 1)
    ψ
end

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = randn(MPS{T}, L, D, d^2) 
    MPO(ψ)
end  

function is_left_normalized(ψ::MPS)
    for i ∈ 1:length(ψ)
        A = ψ[i]
        DD = size(A, 3)
    
        @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
        I(DD) ≈ Id ? () : return false
    end  
    true
end

function is_right_normalized(ϕ::MPS)   
    for i ∈ 1:length(ϕ)
        B = ϕ[i]
        DD = size(B, 1)

        @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        I(DD) ≈ Id ? () : return false
    end 
    true
end

function _verify_square(ψ::AbstractMPS)
    arr = [size(A, 2) for A ∈ ψ]
    @assert isqrt.(arr) .^ 2 == arr "Incorrect MPS dimensions"
end

function verify_physical_dims(ψ::AbstractMPS, dims::NTuple)
    for i ∈ 1:length(ψ)
        @assert size(ψ[i], 2) == dims[i] "Incorrect physical dim at site $(i)." 
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

function Base.show(::IO, ψ::AbstractMPS)
    L = length(ψ)
    dims = [size(A) for A in ψ]

    @info "Matrix product state on $L sites:" 
    _show_sizes(dims)
    println("   ")
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