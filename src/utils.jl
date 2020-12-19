export idx, ising, proj
export HadamardMPS, rq
export all_states, local_basis
export enum

using Base.Cartesian
import Base.Prehashed

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

@generated function _unique_dims(A::AbstractArray{T,N}, dim::Integer) where {T,N}
    quote
        1 <= dim <= $N || return copy(A)
        hashes = zeros(UInt, axes(A, dim))

        # Compute hash for each row
        k = 0
        @nloops $N i A d->(if d == dim; k = i_d; end) begin
            @inbounds hashes[k] = hash(hashes[k], hash((@nref $N A i)))
        end

        # Collect index of first row for each hash
        uniquerow = similar(Array{Int}, axes(A, dim))
        firstrow = Dict{Prehashed,Int}()
        for k = axes(A, dim)
            uniquerow[k] = get!(firstrow, Prehashed(hashes[k]), k)
        end
        uniquerows = collect(values(firstrow))

        # Check for collisions
        collided = falses(axes(A, dim))
        @inbounds begin
            @nloops $N i A d->(if d == dim
                k = i_d
                j_d = uniquerow[k]
            else
                j_d = i_d
            end) begin
                if (@nref $N A j) != (@nref $N A i)
                    collided[k] = true
                end
            end
        end

        if any(collided)
            nowcollided = similar(BitArray, axes(A, dim))
            while any(collided)
                # Collect index of first row for each collided hash
                empty!(firstrow)
                for j = axes(A, dim)
                    collided[j] || continue
                    uniquerow[j] = get!(firstrow, Prehashed(hashes[j]), j)
                end
                for v in values(firstrow)
                    push!(uniquerows, v)
                end

                # Check for collisions
                fill!(nowcollided, false)
                @nloops $N i A d->begin
                    if d == dim
                        k = i_d
                        j_d = uniquerow[k]
                        (!collided[k] || j_d == k) && continue
                    else
                        j_d = i_d
                    end
                end begin
                    if (@nref $N A j) != (@nref $N A i)
                        nowcollided[k] = true
                    end
                end
                (collided, nowcollided) = (nowcollided, collided)
            end
        end

        (@nref $N A d->d == dim ? sort!(uniquerows) : (axes(A, d)))
    end
end