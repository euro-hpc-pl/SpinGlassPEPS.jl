
# gauges.jl: This file provides basic functions to optimize gauges for the PEPS network. CUDA is supported.

export optimize_gauges_for_overlaps!!, overlap_density_matrix

function update_rq!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # QMps tensors are stored as [left, right, physical].  RQ groups the
    # physical and right indices so that the returned factor can be absorbed
    # into the right bond of the preceding tensor.
    ATP = permutedims(AT, (1, 3, 2))
    ATR = reshape(ATP, size(ATP, 1), size(ATP, 2) * size(ATP, 3))
    RT, QT = rq_fact(ATR; toGPU = ψ.onGPU)
    RT ./= maximum(abs.(RT))
    ATP = reshape(QT, size(QT, 1), size(AT, 3), size(AT, 2))
    ψ[i] = permutedims(ATP, (1, 3, 2))
    RT
end


function update_qr!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # QR groups the left and physical indices so that the returned factor can
    # be absorbed into the left bond of the following tensor.
    ATP = permutedims(AT, (1, 3, 2))
    ATR = reshape(ATP, size(ATP, 1) * size(ATP, 2), size(ATP, 3))
    QT, RT = qr_fact(ATR; toGPU = ψ.onGPU)
    RT ./= maximum(abs.(RT))
    ATP = reshape(QT, size(AT, 1), size(AT, 3), size(QT, 2))
    ψ[i] = permutedims(ATP, (1, 3, 2))
    RT
end


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
        @tensor ρ_t[r, s] := T[i, j, r] * conj(T)[i, j, s]
        @tensor ρ_b[r, s] := B[i, j, r] * conj(B)[i, j, s]

        dρ_b, dρ_t = diag.((ρ_b, ρ_t))
        K = (dρ_b .< tol) .|| (dρ_t .< tol)
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        X_inv = 1 ./ X
        gauges[i] .*= X  # update

        RT = update_qr!(ψ_top, T .* reshape(X, 1, 1, :), i)
        RB = update_qr!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)
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

        @tensor T[a, b, c] := T[a, s, c] * RT[s, b]
        @tensor B[a, b, c] := B[a, s, c] * RB[s, b]
        @tensor ρ_t[r, s] := T[i, j, r] * conj(T)[i, j, s]
        @tensor ρ_b[r, s] := B[i, j, r] * conj(B)[i, j, s]

        dρ_b, dρ_t = diag.((ρ_b, ρ_t))
        K = (dρ_b .< tol) .|| (dρ_t .< tol)
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        X_inv = 1 ./ X
        gauges[i] .*= X # update

        RT = update_rq!(ψ_top, T .* reshape(X, 1, 1, :), i)
        RB = update_rq!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)
    end
end

function optimize_gauges_for_overlaps!!(
    ψ_top::QMps{T},
    ψ_bot::QMps{T},
    tol = 1E-8,
    max_sweeps::Int = 4,
) where {T<:Real}
    ψ_top.sites == ψ_bot.sites ||
        throw(ArgumentError("the two QMps objects must have the same sites"))
    ψ_top.onGPU == ψ_bot.onGPU ||
        throw(ArgumentError("the two QMps objects must be on the same device"))
    onGPU = ψ_top.onGPU
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)
    overlap_old = dot(ψ_top, ψ_bot)
    gauges = Dict(i => (onGPU ? CUDA.ones : ones)(T, size(ψ_top[i], 3)) for i ∈ ψ_top.sites)
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
    ψ.sites == ϕ.sites ||
        throw(ArgumentError("the two QMps objects must have the same sites"))
    ψ.onGPU == ϕ.onGPU ||
        throw(ArgumentError("the two QMps objects must be on the same device"))
    position = findfirst(==(k), ψ.sites)
    isnothing(position) && throw(ArgumentError("site $k is not present in the QMps"))
    C = _overlap_forward(ϕ, ψ, position)
    D = _overlap_backwards(ϕ, ψ, position)
    A, B = ψ[k], ϕ[k]
    @tensor E[x, y] := C[b, a] * conj(B)[b, β, x] * A[a, α, y] * D[β, α]
end

function _overlap_forward(ϕ::QMps{T}, ψ::QMps{T}, position::Int) where {T<:Real}
    C = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for site_index ∈ 1:(position-1)
        i = ψ.sites[site_index]
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) C[x, y] := conj(B)[β, x, σ] * C[β, α] * A[α, y, σ]
    end
    C
end

function _overlap_backwards(ϕ::QMps{T}, ψ::QMps{T}, position::Int) where {T<:Real}
    D = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for site_index ∈ length(ψ.sites):-1:(position+1)
        i = ψ.sites[site_index]
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) D[x, y] := conj(B)[x, β, σ] * D[β, α] * A[y, α, σ]
    end
    D
end
