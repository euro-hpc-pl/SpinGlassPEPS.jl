for (T, N) in ((:MPO, 4), (:MPS, 3))
    @eval begin
        export $T

        struct $T{T <: AbstractArray{<:Number, $N}}
            tensors::Vector{T}
        
            function $T{T}(L::Int) where {T}
                new(Vector{T}(undef, L))
            end
        
            $T{T}(v::Vector{T}) where {T} = new(v)
        end
        
        # consturctors
        $T(::Type{T}, ::Type{S}, L::Int) where {T<:AbstractArray, S<:Number} = $T{T{S, $N}}(L)

        Base.:(==)(ϕ::$T, ψ::$T) = ψ.tensors == ϕ.tensors
        Base.:(≈)(ϕ::$T, ψ::$T)  = ψ.tensors ≈ ϕ.tensors

        Base.eltype(::Type{$T{T}}) where {T} = eltype(T)

        Base.getindex(ψ::$T, i::Int) = getindex(ψ.tensors, i)
        Base.setindex!(ψ::$T, A::AbstractArray{<:Number, $N}, i::Int) = ψ.tensors[i] = A
        Base.iterate(ψ::$T) = iterate(ψ.tensors)
        Base.iterate(ψ::$T, state) = iterate(ψ.tensors, state)
        Base.lastindex(ψ::$T) = lastindex(ψ.tensors)

        Base.length(mps::$T) = length(mps.tensors)
        Base.size(ψ::$T) = (length(ψ.tensors), )

        Base.copy(ψ::$T{T}) where {T} = $T{T}(copy(ψ.tensors))
    end
end

function MPO(ψ::MPS{T}) where {T}
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

function MPS(O::MPO{T}) where {T}
    L = length(O)
    ψ = MPS(T.name.wrapper, eltype(T), L)

    for i ∈ 1:L
        W = O[i]
        @cast A[x, (σ, η), y] := W[x, σ, y, η]
        ψ[i] = A     
    end 
    ψ
end  

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = MPS{T}(L)
    S = eltype(T)
    ψ[1] = randn(S, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = randn(S, D, d, D)
    end
    ψ[end] = randn(S, D, d, 1)
    ψ
end

function Base.randn(::Type{MPO{T}}, L::Int, D::Int, d::Int) where {T}
    S = newdim(T, 3)
    ψ = randn(MPS{S}, L, D, d^2) 
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