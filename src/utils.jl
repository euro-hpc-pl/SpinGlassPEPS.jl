export idx, ising, proj
export HadamardMPS, rq
export all_states, local_basis, enum, state_to_ind, rank_vec
export @state

using Base.Cartesian
import Base.Prehashed

function reshape_row(A::AbstractArray{T}, dims::Tuple) where {T <: Number}
    ord = reverse(1:length(dims))
    A = reshape(A, reverse(dims))
    permutedims(A, ord)
end

enum(vec) = Dict(v => i for (i, v) ∈ enumerate(vec))

idx(σ::Int) = (σ == -1) ? 1 : σ + 1
_σ(idx::Int) = (idx == 1) ? -1 : idx - 1

@inline state_to_ind(::AbstractArray, ::Int, i) = i
@inline function state_to_ind(a::AbstractArray, k::Int, σ::State)
    n = length(σ)
    if n == 1 return idx(σ[1]) end
    d = size(a, k)
    base = Int(d ^ (1/n))
    ind = idx.(σ) .- 1
    i = sum(l*base^(j-1) for (j, l) ∈ enumerate(reverse(ind)))
    i + 1
end

function process_ref(ex)
    n = length(ex.args)
    args = Vector(undef, n)
    args[1] = ex.args[1]
    for i=2:length(ex.args)
        args[i] = :(state_to_ind($(ex.args[1]), $(i-1), $(ex.args[i])))
    end
    rex = Expr(:ref)
    rex.args = args
    rex
end

macro state(ex)
    if ex.head == :ref
        rex = process_ref(ex)
    elseif ex.head == :(=) || ex.head == Symbol("'")
        rex = copy(ex)
        rex.args[1] = process_ref(ex.args[1])
    else
        error("Not supported operation: $(ex.head)")
    end
    esc(rex)
end

LinearAlgebra.I(ψ::AbstractMPS, i::Int) = I(size(ψ[i], 2))

local_basis(d::Int) = union(-1, 1:d-1)
local_basis(ψ::AbstractMPS, i::Int) = local_basis(size(ψ[i], 2))

function proj(state, dims::Union{Vector, NTuple})
    P = Matrix{Float64}[] 
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function all_states(rank::Union{Vector, NTuple})
    basis = [local_basis(r) for r ∈ rank]
    product(basis...)
end

function HadamardMPS(rank::Union{Vector, NTuple})
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

function Base.LinearIndices(m::Int, n::Int, origin::Symbol=:NW)
    @assert origin ∈ (:NW, :WN, :NE, :EN, :SE, :ES, :SW, :WS)

    ind = Dict()
    if origin == :NW
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (i - 1) * n + j) end
    elseif origin == :WN
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (j - 1) * n + i) end
    elseif origin == :NE
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (i - 1) * n + (n + 1 - j)) end
    elseif origin == :EN
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (j - 1) * n + (n + 1 - i)) end
    elseif origin == :SE
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (m - i) * n + (n + 1 - j)) end
    elseif origin == :ES
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (m - j) * n + (n + 1 - i)) end
    elseif origin == :SW
        for i ∈ 1:m, j ∈ 1:n push!(ind, (i, j) => (m - i) * n + j) end
    elseif origin == :WS
        for i ∈ 1:n, j ∈ 1:m push!(ind, (i, j) => (m - j) * n + i) end
    end

    if origin ∈ (:NW, :NE, :SE, :SW)
        i_max, j_max = m, n
    else 
        i_max, j_max = n, m
    end

    for i ∈ 0:i_max+1
        push!(ind, (i, 0) => 0)
        push!(ind, (i, j_max + 1) => 0)
    end

    for j ∈ 0:j_max+1
        push!(ind, (0, j) => 0)
        push!(ind, (i_max + 1, j) => 0)
    end

    ind, i_max, j_max
end

@generated function unique_dims(A::AbstractArray{T,N}, dim::Integer) where {T,N}
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
                for v ∈ values(firstrow)
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

        (@nref $N A d->d == dim ? sort!(uniquerows) : (axes(A, d))), indexin(uniquerow, uniquerows)
    end
end

function rank_vec(ig::MetaGraph)
    rank = get_prop(ig, :rank)
    L = get_prop(ig, :L)
    Int[get(rank, i, 1) for i=1:L]
end