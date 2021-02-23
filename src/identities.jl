export IdentityMPO, IdentityMPS
struct IdentityMPS{T <: Number} <: AbstractMPS{T} end
struct IdentityMPO{T <: Number} <: AbstractMPO{T} end

const IdentityMPSorMPO = Union{IdentityMPO, IdentityMPS}

@inline Base.getindex(::IdentityMPS{T}, ::Int) where {T} = ones(T, 1, 1, 1)
@inline Base.getindex(::IdentityMPO{T}, ::Int) where {T} = ones(T, 1, 1, 1, 1)

MPS(::UniformScaling{T}) where {T} = IdentityMPS{T}()
MPO(::UniformScaling{T}) where {T} = IdentityMPO{T}()

LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPO) = O
LinearAlgebra.dot(::IdentityMPO, O::AbstractMPO) = O
Base.length(::IdentityMPSorMPO) = Inf

function LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPS)
    L = length(O)
    T = eltype(O)
    ψ = MPS(T, L)
    for i ∈ eachindex(ψ)
        B = O[i]
        @reduce A[x, σ, y] |= sum(η) B[x, σ, y, η]
        ψ[i] = A
    end
    ψ
end

LinearAlgebra.dot(::IdentityMPO, ψ::AbstractMPS) = ψ
LinearAlgebra.dot(ψ::AbstractMPS, ::IdentityMPO) = ψ

function Base.show(::IO, ψ::IdentityMPSorMPO)
    @info "Trivial matrix product state" 
    println("   ")
end