
# linear_algebra_ext.jl: This file provides basic functions to perform custom SVD, and QR.
#                        CPU inputs use LAPACK; CuMatrix inputs stay on the device (CUSOLVER).

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
    # maximum(Σ) == Σ[1] (sorted), but works on the GPU without scalar indexing
    δ = min(Dcut, sum(Σ .> maximum(Σ) * max(eps(), tol)))
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

# GPU methods: factorize on the device via CUSOLVER instead of round-tripping
# GPU -> CPU LAPACK -> GPU on every site of every sweep. Below the element
# threshold the CUSOLVER launch + Q-materialization overhead loses to LAPACK
# plus the PCIe round trip (measured on RTX 5080: crossover between 8K and
# 32K elements), so small matrices keep the CPU path.
const QR_GPU_MIN_ELEMENTS = 2^15

function qr_fact(
    M::CuMatrix{T},
    Dcut::Int = typemax(Int),
    tol::T = eps(T);
    toGPU::Bool = true,
    kwargs...,
) where {T<:Real}
    if length(M) < QR_GPU_MIN_ELEMENTS
        return qr_fact(Array(M), Dcut, tol; toGPU = toGPU, kwargs...)
    end
    F = qr(M)
    ϕ = phase(diag(F.R); atol = tol)
    q = CuMatrix(F.Q) * Diagonal(ϕ)
    r = ϕ .* F.R
    if Dcut < size(q, 2)
        U, Σ, V = svd_fact(r, Dcut, tol, kwargs...)
        q, r = q * U, Σ .* V'
    end
    toGPU ? (q, r) : (Array(q), Array(r))
end

function rq_fact(
    M::CuMatrix{T},
    Dcut::Int = typemax(Int),
    tol::T = eps(T);
    toGPU::Bool = true,
    kwargs...,
) where {T<:Real}
    if length(M) < QR_GPU_MIN_ELEMENTS
        return rq_fact(Array(M), Dcut, tol; toGPU = toGPU, kwargs...)
    end
    q, r = qr_fact(CuMatrix(M'), Dcut, tol; toGPU = true, kwargs...)
    rt, qt = copy(r'), copy(q')
    toGPU ? (rt, qt) : (Array(rt), Array(qt))
end
