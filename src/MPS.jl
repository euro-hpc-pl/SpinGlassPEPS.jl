export MPS

struct MPS{T <: AbstractArray{<:Number, 3}}
    tensors::Vector{T}

    function MPS{T}(L::Int) where {T}
        new(Vector{T}(undef, L))
    end

    MPS{T}(v::Vector{T}) where {T} = new(v)
end

# consturctors
MPS(::Type{T}, ::Type{S}, L::Int) where {T<:AbstractArray, S<:Number} = MPS{T{S, 3}}(L)


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

# length, comparison, element types

Base.:(==)(ϕ::MPS, ψ::MPS) = ψ.tensors == ϕ.tensors
Base.:(≈)(ϕ::MPS, ψ::MPS)  = ψ.tensors ≈ ϕ.tensors

Base.eltype(::Type{MPS{T}}) where {T} = eltype(T)

Base.getindex(ψ::MPS, i::Int) = getindex(ψ.tensors, i)
Base.setindex!(ψ::MPS, A::AbstractArray{<:Number, 3}, i::Int) = ψ.tensors[i] = A
Base.iterate(ψ::MPS) = iterate(ψ.tensors)
Base.iterate(ψ::MPS, state) = iterate(ψ.tensors, state)
Base.lastindex(ψ::MPS) = lastindex(ψ.tensors)

Base.length(mps::MPS) = length(mps.tensors)
Base.size(ψ::MPS) = (length(ψ.tensors), )

Base.copy(ψ::MPS{T}) where {T} = MPS{T}(copy(ψ.tensors))

# printing

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
