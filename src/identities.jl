export IdentityMPO, IdentityMPS
struct IdentityMPS{T <: Number, S <: AbstractArray} <: AbstractMPS{T} end
struct IdentityMPO{T <: Number, S <: AbstractArray} <: AbstractMPO{T} end
IdentityMPS() = IdentityMPS{Float64, Array}()
IdentityMPO() = IdentityMPO{Float64, Array}()

IdentityMPS(::Type{T}) where {T <: AbstractArray} = IdentityMPS{Float64, T}
IdentityMPO(::Type{T}) where {T <: AbstractArray} = IdentityMPO{Float64, T}

IdentityMPS(::Type{S}, ::Type{T}) where {S <: Number, T <: AbstractArray} = IdentityMPS{S, T}
IdentityMPO(::Type{S}, ::Type{T}) where {S <: Number, T <: AbstractArray} = IdentityMPO{S, T}

const IdentityMPSorMPO = Union{IdentityMPO, IdentityMPS}

@inline function Base.getindex(::IdentityMPS{S, T}, ::Int) where {S, T}
    ret = similar(T{S}, (1, 1, 1))
    ret[1] = one(S)
    ret
end

@inline function Base.getindex(::IdentityMPO{S, T}, ::Int) where {S, T}
    ret = similar(T{S}, (1, 1, 1, 1))
    ret[1] = one(S)
    ret
end

LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPO) = O
LinearAlgebra.dot(::IdentityMPO, O::AbstractMPO) = O
Base.length(::IdentityMPSorMPO) = Inf

function LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPS)
    L = length(O)
    T = eltype(O)
    ψ = MPS(T, L) #FIXME: this will fail with specialized MPS types
    for i ∈ eachindex(ψ)
        B = O[i]
        @reduce A[x, σ, y] |= sum(η) B[x, σ, y, η]
        ψ[i] = A
    end
    ψ
end

LinearAlgebra.dot(::IdentityMPO, ψ::AbstractMPS) = ψ
LinearAlgebra.dot(ψ::AbstractMPS, ::IdentityMPO) = ψ

function Base.show(io::IO, ::IdentityMPSorMPO)
    println(io, "Trivial matrix product state")
    println(io, "   ")
end
