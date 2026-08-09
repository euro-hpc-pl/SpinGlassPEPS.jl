

# variational.jl: This file provides basic functions to perform variational compression for MPS.
#                 If the MPS is moved to the GPU, its compression will be performed on the device.

export variational_compress!, variational_sweep!

function variational_compress!(
    bra::QMps{T},
    mpo::QMpo{T},
    ket::QMps{T},
    tol = 1E-10,
    max_sweeps::Int = 4,
    kwargs...,
) where {T<:Real}
    @assert is_left_normalized(bra)
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_0, negative = measure_env(env, last(env.bra.sites))
    if negative
        env.bra[last(env.bra.sites)] .*= -1
    end

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env; kwargs...)
        _right_sweep_var!(env; kwargs...)
        overlap, negative = measure_env(env, last(env.bra.sites))
        if negative
            env.bra[last(env.bra.sites)] .*= -1
        end
        Δ = abs(overlap_0 - overlap)
        # Per-sweep convergence trace: debug, not info. It fires once per
        # variational sweep per row, so at info level it buries anything else the
        # caller prints (and a warm-started β ladder runs it on every row).
        @debug "Convergence" Δ
        if Δ < tol
            return overlap, env
        else
            overlap_0 = overlap
        end
    end
    overlap, env
end

function _left_sweep_var!(env::Environment; kwargs...)
    for site ∈ reverse(env.bra.sites)
        _left_sweep_var_site!(env, site; kwargs...)
    end
end

function _left_sweep_var_site!(env::Environment, site::Site; kwargs...)
    toGPU = env.ket.onGPU && env.mpo.onGPU && env.bra.onGPU
    update_env_right!(env, site)
    A = project_ket_on_bra(env, site)
    B = reshape(A, size(A, 1), size(A, 2) * size(A, 3))
    _, Q = rq_fact(B; toGPU = toGPU, kwargs...)
    C = reshape(Q, size(Q, 1), size(Q, 2) ÷ size(A, 3), size(A, 3))
    env.bra[site] = C
    clear_env_containing_site!(env, site)
end

function _right_sweep_var!(env::Environment; kwargs...)
    for site ∈ env.bra.sites
        _right_sweep_var_site!(env, site; kwargs...)
    end
end

function _right_sweep_var_site!(env::Environment, site::Site; kwargs...)
    toGPU = env.ket.onGPU && env.mpo.onGPU && env.bra.onGPU
    update_env_left!(env, site)
    A = project_ket_on_bra(env, site)
    B = permutedims(A, (1, 3, 2))  # [l, t, r]
    B = reshape(B, size(B, 1) * size(B, 2), size(B, 3))
    Q, _ = qr_fact(B; toGPU = toGPU, kwargs...)
    C = reshape(Q, size(Q, 1) ÷ size(A, 3), size(A, 3), size(Q, 2))
    C = permutedims(C, (1, 3, 2))  # [l, r, t]
    env.bra[site] = C
    clear_env_containing_site!(env, site)
end

# TODO those 2 functions are to be removed eventually
function variational_sweep!(
    bra::QMps{T},
    mpo::QMpo{T},
    ket::QMps{T},
    ::Val{:left};
    kwargs...,
) where {T<:Real}
    _right_sweep_var!(Environment(bra, mpo, ket); kwargs...)
end

function variational_sweep!(
    bra::QMps{T},
    mpo::QMpo{T},
    ket::QMps{T},
    ::Val{:right};
    kwargs...,
) where {T<:Real}
    _left_sweep_var!(Environment(bra, mpo, ket); kwargs...)
end
