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