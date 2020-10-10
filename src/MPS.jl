export MPS

# from https://github.com/MasonProtter/MatrixProductStates.jl/blob/v0.1/src/MPS.jl
struct MPS{T <: AbstractArray{<:Number, 3}}
    tensors::Vector{T}
end

# consturctors

function MPS(vs::Vector{T}) where {T <: AbstractVector{<:Number}}
    l = length(vs)

    tensrs = _create_empty(T, l)
    for i in 1:l
        tensrs[i] = reshape(copy(vs[i]), 1, 1, :)
    end

    MPS(tensrs)
end

@inline _create_empty(::Type{T}, l) where {T <: AbstractVector{S}} where {S <: Number} = Vector{Array{S, 3}}(undef, l)
@inline _create_empty(::Type{T}, l) where {T <: CuVector{S}} where {S <: Number} = Vector{CuArray{S, 3}}(undef, l)

MPS(v::AbstractVector{T}, l::Int) where {T <: Number} = MPS([v for _ in 1:l])

function MPS(::Type{T}, vs::Vector{S}) where {T <: AbstractVector{<:Number}, S <: AbstractVector{<:Number}}
    MPS(convert(Vector{T}, vs))
end

function MPS(::Type{T}, v::AbstractVector{S}, l::Int) where {T <: AbstractVector{<:Number}, S <: Number}
    cv = convert(T, v)
    MPS(cv, l)
end

# length, comparison, element types

Base.length(mps::MPS) = length(mps.tensors)
Base.:(==)(ψ::MPS, ϕ::MPS) = ψ.tensors == ϕ.tensors
Base.isapprox(ψ::MPS, ϕ::MPS) = isapprox(ψ.tensors, ϕ.tensors)
Base.eltype(::Type{MPS{T}}) where {T <: AbstractArray{S, 3}} where {S <: Number} = S
Base.size(ψ::MPS) = (length(ψ.tensors), )
Base.getindex(ψ::MPS, i::Int) = getindex(ψ.tensors, i)

Base.:(*)(ψ::MPS, x::Number) = MPS(ψ.tensors .* x)
Base.:(*)(x::Number, ψ::MPS) = ψ * x
Base.:(/)(ψ::MPS, x::Number) = MPS(ψ.tensors ./ x)
Base.copy(ψ::MPS) = MPS(copy(ψ.tensors))

function Base.adjoint(ψ::MPS{T}) where {T <: AbstractArray{S, 3}} where {S <: Number}
    Adjoint{S, MPS{T}}(ψ)
end

#TODO: this should use views
function Base.getindex(ψ::Adjoint{S, MPS{T}}, args...) where {T <: AbstractArray{S, 3}} where {S <: Number}
    out = getindex(reverse(ψ.parent.tensors), args...)
    permutedims(conj.(out), (2, 1, 3))
end

Base.size(ψ::Adjoint{S, MPS{T}}) where {T <: AbstractArray{S, 3}} where {S <: Number} = (1, length(ψ.parent[1]))
adjoint_tensors(ψ::MPS) = reverse(conj.(permutedims.(ψ.tensors, [(2, 1, 3)])))

function Base.:(*)(ψ′::Adjoint{S, MPS{T}}, ϕ::MPS{T}) where {T <: AbstractArray{S, 3}} where {S <: Number}
    ψ = ψ′.parent

    M   = ϕ.tensors[1]
    M̃dg = dg(ψ.tensors[1])

    @tensor cont[b₁, a₁] := M̃dg[b₁, 1, σ₁] * M[1, a₁, σ₁]

    l = length(ϕ)
    for i=2:l-1
        M   = ϕ.tensors[i]
        M̃dg = dg(ψ.tensors[i])

        @tensor cont[bᵢ, aᵢ] := M̃dg[bᵢ, bᵢ₋₁, σᵢ] * cont[bᵢ₋₁, aᵢ₋₁] * M[aᵢ₋₁, aᵢ, σᵢ]
    end
    M   = ϕ.tensors[l]
    M̃dg = dg(ψ.tensors[l])
    
    @tensor M̃dg[1, bᴸ⁻¹, σᴸ] * cont[bᴸ⁻¹, aᴸ⁻¹] * M[aᴸ⁻¹, 1, σᴸ]
end

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

#printing

function Base.show(io::IO, ::MIME"text/plain", ψ::MPS)
    l = length(ψ)
    d = length(ψ.tensors[2][1, 1, :])
    bonddims = [size(ψ[i][:, :, 1]) for i in 1:l]
    println(io, "Matrix product state on $l sites")
    _show_mps_dims(io, l, d, bonddims)
end

function Base.show(ψ::MPS)
    l = length(ψ)
    d = length(ψ.tensors[2][1, 1, :])
    bonddims = [size(ψ[i][:, :, 1]) for i in 1:l]
    println("Matrix product state on $l sites")
    _show_mps_dims(l, d, bonddims)
end

function _show_mps_dims(io::IO, l, d, bonddims)
    println(io, "  Physical dimension: $d")
    print(io, "  Bond dimensions:   ")
    if l > 8
        for i in 1:8
            print(io, bonddims[i], " × ")
        end
        print(io, " ... × ", bonddims[l])
    else
        for i in 1:(l-1)
            print(io, bonddims[i], " × ")
        end
        print(io, bonddims[l])
    end
end

function Base.show(io::IO, ψ::MPS)
    l = length(ψ)
    print(io, "MPS on $l sites")
end

function Base.show(io::IO, ::MIME"text/plain", ψ::Adjoint{S, MPS{T}}) where {T <: AbstractArray{S, 3}} where {S <: Number}
    d = length(ψ.parent[2][1, 1, :])
    l = length(ψ)
    bonddims = reverse([reverse(size(ψ.parent[i][:, :, 1])) for i in 1:l])
    println(io, "Adjoint matrix product state on $l sites")
    _show_mps_dims(io, l, d, bonddims)
end

function Base.show(io::IO, ψ::Adjoint{S, MPS{T}}) where {T <: AbstractArray{S, 3}} where {S <: Number}
    l = length(ψ)
    print(io, "Adjoint MPS on $l sites")
end

