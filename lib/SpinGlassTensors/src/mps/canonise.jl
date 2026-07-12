
# canonise.jl: This file provides basic function to left / right truncate / canonise MPS. CUDA is supported.

export canonise!, truncate!, canonise_truncate!, measure_spectrum


function measure_spectrum(ψ::QMps{T}) where {T<:Real}
    # Assume that ψ is left_canonical
    @assert is_left_normalized(ψ)
    R = ones(T, 1, 1)
    schmidt = Dict() # {Site =>AbstractArray}
    for i ∈ reverse(ψ.sites)
        B = permutedims(Array(ψ[i]), (1, 3, 2)) # [x, σ, α]
        @tensor M[x, σ, y] := B[x, σ, α] * R[α, y]
        # @cast M[x, (σ, y)] := M[x, σ, y] TODO: restore when deps merged
        M = reshape(M, :, size(M, 2) * size(M, 3))
        Dcut, tolS = 100000, zero(T)
        U, S, _ = svd_fact(Array(M), Dcut, tolS)
        push!(schmidt, i => S)
        R = U * Diagonal(S)
    end
    schmidt
end



function truncate!(
    ψ::QMps{T},
    s::Symbol,
    Dcut::Int = typemax(Int),
    tolS::T = eps(T);
    kwargs...,
) where {T<:Real}
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ; kwargs...)
        _left_sweep!(ψ, Dcut, tolS; kwargs...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, tolS; kwargs...)
    end
end

canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

function canonise_truncate!(
    ψ::QMps{T},
    type::Symbol,
    Dcut::Int = typemax(Int),
    tolS::T = eps(T);
    kwargs...,
) where {T<:Real}
    if type == :right
        _left_sweep!(ψ, Dcut, tolS; kwargs...)
    elseif type == :left
        _right_sweep!(ψ, Dcut, tolS; kwargs...)
    else
        throw(ArgumentError("Wrong canonization type $type"))
    end
end

function _right_sweep!(
    ψ::QMps{T},
    Dcut::Int = typemax(Int),
    tolS::T = eps(T);
    kwargs...,
) where {T<:Real}
    R = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @tensor M[x, y, σ] := R[x, α] * A[α, y, σ]
        M = permutedims(M, (3, 1, 2))  # [σ, x, y]
        # @cast M[(σ, x), y] := M[σ, x, y] TODO: restore when deps merged
        M = reshape(M, size(M, 1) * size(M, 2), :)
        Q, R = qr_fact(M, Dcut, tolS; toGPU = ψ.onGPU, kwargs...)
        R ./= maximum(abs.(R))
        # @cast A[σ, x, y] := Q[(σ, x), y] (σ ∈ 1:size(A, 3)) TODO: restore when deps merged
        A = reshape(Q, size(A, 3), size(Q, 1) ÷ size(A, 3), size(Q, 2))
        ψ[i] = permutedims(A, (2, 3, 1))  # [x, y, σ]
    end
end

function _left_sweep!(
    ψ::QMps{T},
    Dcut::Int = typemax(Int),
    tolS::T = eps(T);
    kwargs...,
) where {T<:Real}
    R = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for i ∈ reverse(ψ.sites)
        B = permutedims(ψ[i], (1, 3, 2)) # [x, σ, α]
        @tensor M[x, σ, y] := B[x, σ, α] * R[α, y]
        # @cast M[x, (σ, y)] := M[x, σ, y]
        M = reshape(M, size(M, 1), size(M, 2) * size(M, 3))
        R, Q = rq_fact(M, Dcut, tolS; toGPU = ψ.onGPU, kwargs...)
        R ./= maximum(abs.(R))
        # @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        B = reshape(Q, size(Q, 1), size(B, 2), size(Q, 2) ÷ size(B, 2))

        ψ[i] = permutedims(B, (1, 3, 2))
    end
end
