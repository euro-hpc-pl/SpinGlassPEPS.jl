export MPS

struct MPS{T <: AbstractArray{<:Number, 3}}
    tensors::Vector{T}
end

# consturctors

function Base.randn(::Type{MPS{T}}, L::Int, D::Int, d::Int) where T <: AbstractArray{<:Number, 3}
    tensors = Vector{T}(undef, L)
    S = eltype(T)

    tensors[1] = randn(S, 1, d, D)
    for i ∈ 2:(L-1)
        tensors[i] = randn(S, D, d, D)
    end
    tensors[end] = randn(S, D, d, 1)
    MPS(tensors) 
end

function MPS(ψ::Vector{T}) where T <: AbstractVector{<:Number}
    L = length(ψ)
    tensors = Vector{T}(undef, L)

    for i ∈ 1:L
        tensors[i] = reshape(copy(ψ[i]), 1, :, 1)
    end
    MPS(tensors)
end

# length, comparison, element types

Base.:(==)(ϕ::MPS, ψ::MPS) = ψ.tensors == ϕ.tensors
Base.:(≈)(ϕ::MPS, ψ::MPS)  = ψ.tensors ≈ ϕ.tensors

Base.eltype(::Type{MPS{T}}) where {T <: AbstractArray{S, 3}} where {S <: Number} = S

Base.getindex(ψ::MPS, i::Int) = getindex(ψ.tensors, i)
Base.setindex(ψ::MPS, i::Int) = setindex(ψ.tensors, i)

Base.length(mps::MPS) = length(mps.tensors)
Base.size(ψ::MPS) = (length(ψ.tensors), )

Base.copy(ψ::MPS) = MPS(copy(ψ.tensors))

# printing

function Base.show(ψ::MPS{T}) where T <: AbstractArray{<:Number, 3}
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
