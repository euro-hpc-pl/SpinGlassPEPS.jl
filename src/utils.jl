export idx, ising, proj
export HadamardMPS, rq
export all_states, local_basis

ising(σ::Union{Vector, NTuple}) = 2 .* σ .- 1

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
_σ(idx::Int) = (idx == 1) ? -1 : idx - 1 

local_basis(d::Int) = union(-1, 1:d-1)

function proj(state::T, dims::S) where {T, S <: Union{Vector, NTuple}}
    P = [] 
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
    MPS(fill(1, r) ./ sqrt(r) for r ∈ rank)
end
HadamardMPS(L::Int) = MPS(fill([1., 1.] / sqrt(2), L))

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