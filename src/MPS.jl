export MPS
# from https://github.com/MasonProtter/MatrixProductStates.jl/blob/v0.1/src/MPS.jl
struct MPS{T <: AbstractArray{<:Number, 3}}
    tensors::Vector{T}
end

Base.length(mps::MPS) = length(mps.tensors)

Base.isequal(ψ::MPS, ϕ::MPS) = (isequal(ψ.tensors, ϕ.tensors))
Base.isapprox(ψ::MPS, ϕ::MPS) = isapprox(ψ.tensors, ϕ.tensors)

Base.eltype(::Type{MPS{T}}) = eltype(T)

Base.size(::MPS) = (length(mps.tensors), )

Base.getindex(ψ::MPS, i::Int) = getindex(ψ.tensors, i)

Base.:(*)(ψ::MPS, x::Number) = MPS(ψ.tensors .* x)
Base.:(*)(x::Number, ψ::MPS) = ψ * x
Base.:(/)(ψ::MPS, x::Number) = MPS(ψ.tensors ./ x)
Base.copy(ψ::MPS) = MPS(copy(ψ.tensors))

function MPS(vs::Vector{Vector{T<:Number}})
    L = length(vs)

    tensrs = Vector{Array{T,3}}(undef, L)
    for i in 1:L
        tensrs[i] = reshape(copy(vs[i]), 1, 1, :)
    end

    MPS{L,T}(tensrs)
end

MPS(v::Vector, L) = MPS([v for _ in 1:L])

function Base.show(io::IO, ::MIME"text/plain", ψ::MPS{L, T}) where {L, T}
    d = length(ψ.tensors[2][1, 1, :])
    bonddims = [size(ψ[i][:, :, 1]) for i in 1:L]
    println(io, "Matrix product state on $L sites")
    _show_mps_dims(io, L, d, bonddims)
end

function Base.show(ψ::MPS{L, T}) where {L, T}
    d = length(ψ.tensors[2][1, 1, :])
    bonddims = [size(ψ[i][:, :, 1]) for i in 1:L]
    println("Matrix product state on $L sites")
    _show_mps_dims(L, d, bonddims)
end

function _show_mps_dims(io::IO, L, d, bonddims)
    println(io, "  Physical dimension: $d")
    print(io, "  Bond dimensions:   ")
    if L > 8
        for i in 1:8
            print(io, bonddims[i], " × ")
        end
        print(io, " ... × ", bonddims[L])
    else
        for i in 1:(L-1)
            print(io, bonddims[i], " × ")
        end
        print(io, bonddims[L])
    end
end

function Base.show(io::IO, ψ::MPS{L, T}) where {L, T}
    print(io, "MPS on $L sites")
end

function Base.adjoint(ψ::MPS)
    Adjoint{T, MPS}(ψ)
end

function Base.show(io::IO, ::MIME"text/plain", ψ::Adjoint{T, MPS})
    d = length(ψ.parent[2][1, 1, :])
    bonddims = reverse([reverse(size(ψ.parent[i][:, :, 1])) for i in 1:L])
    println(io, "Adjoint matrix product state on $L sites")
    _show_mps_dims(io, L, d, bonddims)
end

function Base.show(io::IO, ψ::Adjoint{T, MPS})
    print(io, "Adjoint MPS on $L sites")t
end

Base.size(ψ::Adjoint{T, MPS}) = (1, length(ψ.parent[1]))

function Base.getindex(ψ::Adjoint{T, MPS}, args...)
    out = getindex(reverse(ψ.parent.tensors), args...)
    permutedims(conj.(out), (2, 1, 3))
end

adjoint_tensors(ψ::MPS) = reverse(conj.(permutedims.(ψ.tensors, [(2, 1, 3)])))