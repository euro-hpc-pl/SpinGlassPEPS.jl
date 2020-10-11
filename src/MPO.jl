export MPO
struct MPO{T <: AbstractArray{<:Number, 4}}
    tensors::Vector{T}
end

function MPO(W::T, l) where T <:AbstractArray{<:Number, 4}
    l >= 2 || throw(DomainError(l, "At least 2 sites."))

    tensors = Vector{T}(undef, l)
    
    tensors[1] = W[end:end, :, :, :]
    for i in 2:(l-1)
        tensors[i] = W # Matrix
    end
    tensors[l] = W[:, 1:1, :, :] # Column vector.

    MPO(tensors)
end

Base.:(==)(O::MPO, U::MPO) = O.tensors == U.tensors
Base.:(≈)(O::MPO, U::MPO)  = O.tensors ≈ U.tensors
Base.getindex(O::MPO, args...) = getindex(O.tensors, args...)
Base.length(O::MPO) = length(O.tensors)

#contractions

function Base.:(*)(O::MPO, ψ::MPS)
    tensors = copy(ψ.tensors)
    l = length(O)
    for i in 1:l
        W = O.tensors[i]
        M = ψ.tensors[i]

        @reduce N[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ] :=  sum(σ′ᵢ) W[bᵢ₋₁, bᵢ, σᵢ, σ′ᵢ] * M[aᵢ₋₁, aᵢ, σ′ᵢ]
        
        tensors[i] = N
    end
    MPS(tensors)
end

function Base.:(*)(O1::MPO{T}, O2::MPO{T}) where {T}
    tensors = copy(O1.tensors)
    l = length(O1)
    for i in 1:l
        W1 = O1.tensors[i]
        W2 = O2.tensors[i]

        @reduce V[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ, σ′ᵢ] :=  sum(σ′′ᵢ) W1[bᵢ₋₁, bᵢ, σᵢ, σ′′ᵢ] * W2[aᵢ₋₁, aᵢ, σ′′ᵢ, σ′ᵢ]
        
        tensors[i] = V
    end
    MPO{T}(tensors)
end

function Base.:(*)(ψ′::Adjoint{S, MPS{T}}, O::MPO) where {T <: AbstractArray{S, 3}} where {S <: Number}
    ψ = ψ′.parent
    tensors = copy(ψ.tensors)
    Ws = dg.(reverse(O.tensors))
    l = length(O)
    for i in 1:l
        W = Ws[i]
        M = ψ.tensors[i]

        @reduce A[(bᵢ₋₁, aᵢ₋₁), (bᵢ, aᵢ), σᵢ] :=  sum(σ′ᵢ) W[bᵢ₋₁, bᵢ, σᵢ, σ′ᵢ] * M[aᵢ₋₁, aᵢ, σ′ᵢ]
        tensors[i] = A
    end
    adjoint(MPS{T}(tensors))
end

#printing

function Base.show(io::IO, ::MIME"text/plain", O::MPO)
    d = length(O[2][1, 1, 1, :])
    l = length(O)
    bonddims = [size(O[i][:, :, 1, 1]) for i in 1:l]
    println(io, "Matrix product Operator on $l sites")
    _show_mpo_dims(io, l, d, bonddims)
end

function _show_mpo_dims(io::IO, l, d, bonddims)
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

function Base.show(io::IO, O::MPO)
    l = length(O)
    print(io, "MPO on $l sites")
end