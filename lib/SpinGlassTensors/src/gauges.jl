
# gauges.jl: This file provides basic functions to optimize gauges for the PEPS network. CUDA is supported.

export optimize_gauges_for_overlaps!!, overlap_density_matrix

function update_rq!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # @cast ATR[x, (σ, y)] := AT[x, σ, y]
    ATR = reshape(AT, size(AT, 1), size(AT, 2) * size(AT, 3))
    RT, QT = rq_fact(ATR)
    RT ./= maximum(abs.(RT))
    # @cast AT[x, σ, y] := QT[x, (σ, y)] (σ ∈ 1:size(AT, 2))
    AT = reshape(QT, size(QT, 1), size(AT, 2), size(QT, 2) ÷ size(AT, 2))
    ψ[i] = AT
    RT
end

# function update_rq!(ψ::QMps{T}, AT::CuArray{T,3}, i::Site) where {T<:Real}
#     # @cast ATR[x, (σ, y)] := AT[x, σ, y]
#     ATR = reshape(AT, size(AT, 1), size(AT, 2) * size(AT, 3))
#     RT, QT = rq_fact(ATR)
#     RT ./= maximum(abs.(RT))
#     @cast AT[x, σ, y] := QT[x, (σ, y)] (σ ∈ 1:size(AT, 2))
#     ψ[i] = AT
#     RT
# end

function update_qr!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # @cast ATR[(x, σ), y] := AT[x, σ, y]
    ATR = reshape(AT, size(AT, 1) * size(AT, 2), size(AT, 3))
    QT, RT = qr_fact(ATR)
    RT ./= maximum(abs.(RT))
    # @cast AT[x, σ, y] := QT[(x, σ), y] (σ ∈ 1:size(AT, 2))
    AT = reshape(QT, size(QT, 1) ÷ size(AT, 2), size(AT, 2), size(QT, 2))
    ψ[i] = AT
    RT
end

# function update_qr!(ψ::QMps{T}, AT::CuArray{T,3}, i::Site) where {T<:Real}
#     @cast ATR[(x, σ), y] := AT[x, σ, y]
#     QT, RT = qr_fact(ATR)
#     RT ./= maximum(abs.(RT))
#     @cast AT[x, σ, y] := QT[(x, σ), y] (σ ∈ 1:size(AT, 2))
#     ψ[i] = AT
#     RT
# end

function _gauges_right_sweep!!!(
    ψ_top::QMps{R},
    ψ_bot::QMps{R},
    gauges::Dict;
    tol = 1E-12,
) where {R<:Real}
    RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
    RB = copy(RT)
    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
        @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b, dρ_t = diag.((ρ_b, ρ_t))
        K = (dρ_b .< tol) .|| (dρ_t .< tol)
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        X_inv = 1 ./ X
        gauges[i] .*= X  # update

        RT = update_qr!(ψ_top, T .* reshape(X, 1, :, 1), i)
        RB = update_qr!(ψ_bot, B .* reshape(X_inv, 1, :, 1), i)
    end
end

function _gauges_left_sweep!!!(
    ψ_top::QMps{R},
    ψ_bot::QMps{R},
    gauges::Dict;
    tol = 1E-12,
) where {R<:Real}
    RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
    RB = copy(RT)
    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := T[a, b, s] * RT[s, c]
        @tensor B[a, b, c] := B[a, b, s] * RB[s, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b, dρ_t = diag.((ρ_b, ρ_t))
        K = (dρ_b .< tol) .|| (dρ_t .< tol)
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        X_inv = 1 ./ X
        gauges[i] .*= X # update

        RT = update_rq!(ψ_top, T .* reshape(X, 1, :, 1), i)
        RB = update_rq!(ψ_bot, B .* reshape(X_inv, 1, :, 1), i)
    end
end

function optimize_gauges_for_overlaps!!(
    ψ_top::QMps{T},
    ψ_bot::QMps{T},
    tol = 1E-8,
    max_sweeps::Int = 4,
) where {T<:Real}
    onGPU = ψ_top.onGPU && ψ_bot.onGPU
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)
    overlap_old = dot(ψ_top, ψ_bot)
    gauges = Dict(i => (onGPU ? CUDA.ones : ones)(T, size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    #gauges = Dict(i => ones(T, size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    for _ ∈ 1:max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, gauges)
        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        if abs(Δ - one(T)) < tol
            break
        end
    end
    gauges
end

function overlap_density_matrix(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    @assert ψ.sites == ϕ.sites
    C = _overlap_forward(ϕ, ψ, k)
    D = _overlap_backwards(ϕ, ψ, k)
    A, B = ψ[k], ϕ[k]
    @tensor E[x, y] := C[b, a] * conj(B)[b, x, β] * A[a, y, α] * D[β, α]
end

function _overlap_forward(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    C = ϕ.onGPU && ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    i = ψ.sites[1]
    while i < k
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y]
        i += 1
    end
    C
end

function _overlap_backwards(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    D = ϕ.onGPU && ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    i = ψ.sites[end]
    while i > k
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) D[x, y] := conj(B)[x, σ, β] * D[β, α] * A[y, σ, α]
        i -= 1
    end
    D
end
