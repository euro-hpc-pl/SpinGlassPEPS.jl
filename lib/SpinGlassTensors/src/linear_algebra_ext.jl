
# linear_algebra_ext.jl: This file provides basic functions to perform custom SVD, and QR.
#                        Both are calculated on CPU, but can be transferd to GPU if need be.

export rq_fact, qr_fact, svd_fact

@inline phase(d::T; atol = eps()) where {T<:Real} =
    isapprox(d, zero(T), atol = atol) ? one(T) : d / abs(d)
@inline phase(d::AbstractArray; atol = eps()) = map(x -> phase(x; atol = atol), d)

function svd_fact(
    A::AbstractMatrix{T},
    Dcut::Int = typemax(Int),
    tol = eps(T);
    kwargs...,
) where {T<:Real}
    U, Σ, V = svd(A; kwargs...)
    δ = min(Dcut, sum(Σ .> Σ[1] * max(eps(), tol)))
    U, Σ, V = U[:, 1:δ], Σ[1:δ], V[:, 1:δ]
    Σ ./= sqrt(sum(Σ .^ 2))
    ϕ = reshape(phase(diag(U); atol = tol), 1, :)
    U .* ϕ, Σ, V .* ϕ
end


function qr_fact(
    M::AbstractMatrix{T},
    Dcut::Int = typemax(Int),
    tol::T = eps(T);
    toGPU::Bool = true,
    kwargs...,
) where {T<:Real}
    q, r = qr_fix(qr(Array(M); kwargs...))
    if Dcut >= size(q, 2)
        toGPU && return CuArray.((q, r))
        return q, r
    end
    U, Σ, V = svd_fact(r, Dcut, tol, kwargs...)
    toGPU && return CuArray.((q * U, Σ .* V'))
    q * U, Σ .* V'
end


function rq_fact(
    M::AbstractMatrix{T},
    Dcut::Int = typemax(Int),
    tol::T = eps(T);
    toGPU::Bool = true,
    kwargs...,
) where {T<:Real}
    q, r = qr_fact(M', Dcut, tol; toGPU = toGPU, kwargs...)
    toGPU && return CuArray.((r', q'))
    r', q'
end

function qr_fix(QR_fact; tol::T = eps()) where {T<:Real}
    ϕ = phase(diag(QR_fact.R); atol = tol)
    QR_fact.Q * Diagonal(ϕ), ϕ .* QR_fact.R
end
