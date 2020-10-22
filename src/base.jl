export bond_dimension
for (T, N) in ((:MPO, 4), (:MPS, 3))
    @eval begin
        export $T
        abstract type $(Symbol(:Abstract, T)) end
        struct $T{T <: Number} <: $(Symbol(:Abstract, T))
            tensors::Vector{Array{T, $N}}
        
            # $T{T}(v::Vector{Array{T, 3}}) where {T} = new(v)
        end
        
        # consturctors
        # $T{T}(L::Int) where {T} = $T(Vector{Array{T, 3}}(undef, L))
        $T(::Type{T}, L::Int) where {T} = $T(Vector{Array{T, $N}}(undef, L))
        $T(L::Int) = $T(Float64, L)

        Base.:(==)(a::$T, b::$T) = a.tensors == b.tensors
        Base.:(≈)(a::$T, b::$T)  = a.tensors ≈ b.tensors

        Base.eltype(::Type{$T{T}}) where {T} = T

        Base.getindex(a::$T, i::Int) = getindex(a.tensors, i)
        Base.setindex!(a::$T, A::AbstractArray{<:Number, $N}, i::Int) = a.tensors[i] = A
        Base.iterate(a::$T) = iterate(a.tensors)
        Base.iterate(a::$T, state) = iterate(a.tensors, state)
        Base.lastindex(a::$T) = lastindex(a.tensors)

        Base.length(a::$T) = length(a.tensors)
        Base.size(a::$T) = (length(a.tensors), )

        Base.copy(a::$T{T}) where {T} = $T{T}(copy(a.tensors))

        bond_dimension(a::$T) = maixmum(size.(a.tensors, $N))
    end
end

function MPS(vec::Vector{<:Number})
    L = length(vec)
    ψ = MPS(L)
    for i ∈ 1:L
        ψ[i] = reshape(copy(vec[i]), 1, :, 1)
    end    
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

function _verify_square(ψ::MPS)
    arr = [size(A, 2) for A ∈ ψ]
    @assert isqrt.(arr) .^ 2 == arr "Incorrect MPS dimensions"
end

function Base.show(::IO, ψ::MPS)
    L = length(ψ)
    σ_list = [size(ψ[i], 2) for i ∈ 1:L] 
    χ_list = [size(ψ[i][:, 1, :]) for i ∈ 1:L]
 
    println("Matrix product state on $L sites:")
    println("Physical dimensions: ")
    _show_sizes(σ_list)
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
