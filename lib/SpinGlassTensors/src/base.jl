# base.jl: This file defines basic tensor structures to be used with SpinGlassEngine

export Tensor,
    SiteTensor,
    VirtualTensor,
    DiagonalTensor,
    CentralTensor,
    CentralOrDiagonal,
    dense_central

abstract type AbstractSparseTensor{T,N} end

mutable struct SiteTensor{T<:Real,N} <: AbstractSparseTensor{T,N}
    lp::PoolOfProjectors
    loc_exp::AbstractVector{T}
    projs::NTuple{4,Int} # pl, pt, pr, pb
    dims::Dims{N}

    function SiteTensor(lp::PoolOfProjectors, loc_exp, projs::NTuple{4,Vector{Int}})
        T = eltype(loc_exp)
        ks = Tuple(add_projector!(lp, p) for p ∈ projs)
        dims = size.(Ref(lp), ks)
        new{T,4}(lp, loc_exp, ks, dims)
    end

    function SiteTensor(
        lp::PoolOfProjectors,
        loc_exp,
        projs::NTuple{4,Int},
        dims::NTuple{4,Int},
    )
        T = eltype(loc_exp)
        new{T,4}(lp, loc_exp, projs, dims)
    end
end


function mpo_transpose(ten::SiteTensor)
    perm = [1, 4, 3, 2]
    SiteTensor(ten.lp, ten.loc_exp, ten.projs[perm], ten.dims[perm])
end

mutable struct CentralTensor{T<:Real,N} <: AbstractSparseTensor{T,N}
    e11::AbstractMatrix{T}
    e12::AbstractMatrix{T}
    e21::AbstractMatrix{T}
    e22::AbstractMatrix{T}
    dims::Dims{N}

    function CentralTensor(e11, e12, e21, e22)
        s11, s12, s21, s22 = size.((e11, e12, e21, e22))
        @assert s11[1] == s12[1] && s21[1] == s22[1] && s11[2] == s21[2] && s12[2] == s22[2]
        dims = (s11[1] * s21[1], s11[2] * s12[2])
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T,2}(e11, e12, e21, e22, dims)
    end
end

Base.adjoint(M::CentralTensor{R,2}) where {R<:Real} =
    CentralTensor(M.e11', M.e21', M.e12', M.e22')

mpo_transpose(ten::CentralTensor) =
    CentralTensor(permutedims.((ten.e11, ten.e21, ten.e12, ten.e22), Ref((2, 1)))...)

const MatOrCentral{T,N} = Union{AbstractMatrix{T},CentralTensor{T,N}}

# TODO: to be removed eventually
function dense_central(ten::CentralTensor)
    # @cast V[(u1, u2), (d1, d2)] :=
    #     ten.e11[u1, d1] * ten.e21[u2, d1] * ten.e12[u1, d2] * ten.e22[u2, d2]
    a11 = reshape(ten.e11, size(ten.e11, 1), :, size(ten.e11, 2))
    a21 = reshape(ten.e21, :, size(ten.e21, 1), size(ten.e21, 2))
    a12 = reshape(ten.e12, size(ten.e12, 1), 1, 1, size(ten.e12, 2))
    a22 = reshape(ten.e22, 1, size(ten.e22, 1), 1, size(ten.e22, 2))
    V = @__dot__(a11 * a21 * a12 * a22)
    V = reshape(V, size(V, 1) * size(V, 2), size(V, 3) * size(V, 4))
    V ./ maximum(V)
end
dense_central(ten::AbstractArray) = ten

mutable struct DiagonalTensor{T<:Real,N} <: AbstractSparseTensor{T,N}
    e1::MatOrCentral{T,N}
    e2::MatOrCentral{T,N}
    dims::Dims{N}

    function DiagonalTensor(e1, e2)
        dims = (size(e1, 1) * size(e2, 1), size(e1, 2) * size(e2, 2))
        T = promote_type(eltype.((e1, e2))...)
        new{T,2}(e1, e2, dims)
    end
end

mpo_transpose(ten::DiagonalTensor) = DiagonalTensor(mpo_transpose.((ten.e2, ten.e1))...)

mutable struct VirtualTensor{T<:Real,N} <: AbstractSparseTensor{T,N}
    lp::PoolOfProjectors
    con::MatOrCentral{T,2}
    projs::NTuple{6,Int}  # == (p_lb, p_l, p_lt, p_rb, p_r, p_rt)
    dims::Dims{N}

    function VirtualTensor(lp::PoolOfProjectors, con, projs::NTuple{6,Vector{Int}})
        T = eltype(con)
        ks = Tuple(add_projector!(lp, p) for p ∈ projs)
        dims = (
            length(lp, ks[2]),
            size(lp, ks[3]) * size(lp, ks[6]),
            length(lp, ks[5]),
            size(lp, ks[1]) * size(lp, ks[4]),
        )
        new{T,4}(lp, con, ks, dims)
    end

    function VirtualTensor(
        lp::PoolOfProjectors,
        con,
        projs::NTuple{6,Int},
        dims::NTuple{4,Int},
    )
        T = eltype(con)
        new{T,4}(lp, con, projs, dims)
    end
end

mpo_transpose(ten::VirtualTensor) =
    VirtualTensor(ten.lp, ten.con, ten.projs[[3, 2, 1, 6, 5, 4]], ten.dims[[1, 4, 3, 2]])
mpo_transpose(ten::AbstractArray{T,4}) where {T} = permutedims(ten, (1, 4, 3, 2))
mpo_transpose(ten::AbstractArray{T,2}) where {T} = permutedims(ten, (2, 1))

const SparseTensor{T,N} =
    Union{SiteTensor{T,N},VirtualTensor{T,N},CentralTensor{T,N},DiagonalTensor{T,N}}
const Tensor{T,N} = Union{AbstractArray{T,N},SparseTensor{T,N}}
const CentralOrDiagonal{T,N} = Union{CentralTensor{T,N},DiagonalTensor{T,N}}

Base.eltype(ten::Tensor{T,N}) where {T,N} = T
Base.ndims(ten::Tensor{T,N}) where {T,N} = N
Base.size(ten::SparseTensor, n::Int) = ten.dims[n]
Base.size(ten::SparseTensor) = ten.dims
