export idx, ising, proj
export HadamardMPS, rq
export Reshape, all_states

proj(s::Int) = proj(Val(s))
proj(::Val{1}) = [[1., 0.] [0., 0.]]
proj(::Val{-1}) = [[0., 0.] [0., 1.]]
proj(state::Vector{Int}) = proj.(state)

ising(state::Vector{Int}) = 2 .* state .- 1

#idx(s::Int) = idx(Val(s))
#idx(::Val{-1}) = 1
#idx(::Val{1}) = 2

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
_σ(idx::Int) = (idx == 1) ? -1 : idx - 1

function all_states(rank::Union{Vector, NTuple})
    basis = [union(-1, 1:r-1) for r ∈ rank]
    product(basis...)
end 

HadamardMPS(L::Int) = MPS(fill([1., 1.] / sqrt(2), L))
#HadamardMPS(L::Int, d::Int) = MPS(fill([1., 1.] / sqrt(2), L))

function Reshape(A::AbstractArray{T}, dims::Tuple) where {T <: Number}
    ord = reverse(1:length(dims))

    A = reshape(A, reverse(dims))
    permutedims(A, ord)
end 

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