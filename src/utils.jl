export idx, ising, proj
export HadamardMPS, rq
export all_states, local_basis
export enum

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
_σ(idx::Int) = (idx == 1) ? -1 : idx - 1

enum(vec) = Dict(v => i for (i, v) ∈ enumerate(vec))

LinearAlgebra.I(ψ::AbstractMPS, i::Int) = I(size(ψ[i], 2))

local_basis(d::Int) = union(-1, 1:d-1)
local_basis(ψ::AbstractMPS, i::Int) = local_basis(size(ψ[i], 2))

function proj(state, dims::T) where {T <: Union{Vector, NTuple}}
    P = Matrix{Float64}[] 
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function all_states(rank::T) where T <: Union{Vector, NTuple}
    basis = [local_basis(r) for r ∈ rank]
    product(basis...)
end

function HadamardMPS(rank::T) where T <: Union{Vector, NTuple}
    vec = [ fill(1, r) ./ sqrt(r) for r ∈ rank ]
    MPS(vec)
end

HadamardMPS(L::Int) = MPS(fill(2, L))

function LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int, args...)
    fact = pqrfact(M, rank=Dcut, args...)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix(Q, R)
end

function rq(M::AbstractMatrix, Dcut::Int, args...)
    fact = pqrfact(:c, conj.(M), rank=Dcut, args...)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix(Q, R)'
end

function _qr_fix(Q::AbstractMatrix, R::AbstractMatrix)
    d = diag(R)
    ph = d./abs.(d)
    idim = size(R, 1)
    q = Matrix(Q)[:, 1:idim]
    return transpose(ph) .* q
end

function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = psvd(A, rank=Dcut, args...)
    d = diag(U)
    ph = d ./ abs.(d)
    return  U * Diagonal(ph), Σ, V * Diagonal(ph)
end
