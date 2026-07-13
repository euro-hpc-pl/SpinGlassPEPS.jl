export Environment, EnvironmentMixed, left_nbrs_site, right_nbrs_site

abstract type AbstractEnvironment end

mutable struct EnvironmentMixed{T<:Real} <: AbstractEnvironment
    bra::QMps{T}  # mps that is to be optimized
    mpo::QMpo{T}
    ket::QMps{T}
    C::Tensor{T,3}
    site::Any  # position of C is at: site - epsilon  ::Union(Sites, :central)
    env::Dict
    onGPU::Bool

    function EnvironmentMixed(
        bra::QMps{T},
        C::Tensor{T,3},
        mpo::QMpo{T},
        ket::QMps{T};
    ) where {T<:Real}
        onGPU = bra.onGPU && mpo.onGPU && ket.onGPU
        @assert bra.sites == ket.sites && issubset(bra.sites, mpo.sites)
        id3 = onGPU ? CUDA.ones(T, 1, 1, 1) : ones(T, 1, 1, 1)
        id2 = onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
        env0 = Dict{Any,Any}((bra.sites[1], :left) => id2, (bra.sites[end], :right) => id3)
        env = new{T}(bra, mpo, ket, C, last(bra.sites) + 1, env0, onGPU)  #
        update_env_left!.(Ref(env), env.bra.sites)
        env
    end
end

function clear_env_containing_site!(env::EnvironmentMixed, site)
    if site == :central
        delete!(env.env, (env.site, :left))
        delete!(env.env, (left_nbrs_site(env.site, env.ket.sites), :right))
    else
        if site == env.site
            delete!(env.env, (:central, :right))
        else
            delete!(env.env, (left_nbrs_site(site, env.ket.sites), :right))
        end
        rs = right_nbrs_site(site, env.ket.sites)
        if rs == env.site
            delete!(env.env, (:central, :left))
        else
            delete!(env.env, (rs, :left))
        end
    end
end

mutable struct Environment{T<:Real} <: AbstractEnvironment
    bra::QMps{T}  # mps that is to be optimized
    mpo::QMpo{T}
    ket::QMps{T}
    env::Dict
    log_norms::Dict

    function Environment(bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}) where {T<:Real}
        onGPU = bra.onGPU && mpo.onGPU && ket.onGPU
        @assert bra.sites == ket.sites && issubset(bra.sites, mpo.sites)
        id = onGPU ? CUDA.ones(T, 1, 1, 1) : ones(T, 1, 1, 1)
        env0 = Dict((bra.sites[1], :left) => id, (bra.sites[end], :right) => id)
        ln0 = Dict((bra.sites[1], :left) => zero(T), (bra.sites[end], :right) => zero(T))
        env = new{T}(bra, mpo, ket, env0, ln0)
        update_env_left!.(Ref(env), env.bra.sites)
        env
    end
end

function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (right_nbrs_site(site, env.ket.sites), :left))
end

"""
Largest x in sites: x < site
"""
function left_nbrs_site(site::Site, sites)
    ls = filter(i -> i < site, sites)
    isempty(ls) && return -Inf
    maximum(ls)
end

"""
Smallest x in sites: x > site
"""
function right_nbrs_site(site::Site, sites)
    ms = filter(i -> i > site, sites)
    isempty(ms) && return Inf
    minimum(ms)
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S,
    A::S,
    M::T,
    B::S,
) where {S<:AbstractArray{R,3},T<:MpoTensor{R,4}} where {R<:Real}
    for v ∈ M.top
        A = contract_tensor3_matrix(A, v)
    end
    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)
    end
    update_env_left(LE, A, M.ctr, B)
end

function update_env_left!(env::Environment, site::Site)
    site <= first(env.bra.sites) && return
    ls = left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls])
    rs = right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = contract_tensor3_matrix(LL, env.mpo[rs])
        rs = right_nbrs_site(rs, env.mpo.sites)
    end
    nLL = maximum(abs.(LL))
    LL ./= nLL
    push!(env.env, (site, :left) => LL)
    nLL = env.log_norms[(ls, :left)] + log(nLL)
    push!(env.log_norms, (site, :left) => nLL)
end

function update_env_left!(env::EnvironmentMixed{T}, site) where {T}    # site::Union(Sites, :central)
    if site == first(env.bra.sites)
        if env.site == first(env.bra.sites)
            LL = env.onGPU ? CUDA.ones(T, 1, 1, 1) : ones(T, 1, 1, 1)
        else
            LL = env.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
        end
    elseif site == :central
        if env.site == first(env.bra.sites)
            LL = env.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
        else
            ls = left_nbrs_site(env.site, env.bra.sites)
            LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.ket[ls])
            LL ./= maximum(abs.(LL))
        end
    elseif site < env.site
        ls = left_nbrs_site(site, env.bra.sites)
        LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.ket[ls])
        LL ./= maximum(abs.(LL))
    elseif site == env.site
        ls = left_nbrs_site(site, env.bra.sites)
        LL = update_env_left(env.env[(:central, :left)], env.C)
        LL ./= maximum(abs.(LL))
    else
        ls = left_nbrs_site(site, env.bra.sites)
        LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls])
        rs = right_nbrs_site(ls, env.mpo.sites)
        while rs < site
            LL = contract_tensor3_matrix(LL, env.mpo[rs])
            rs = right_nbrs_site(rs, env.mpo.sites)
        end
        LL ./= maximum(abs.(LL))
    end
    push!(env.env, (site, :left) => LL)
end


"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S,
    A::S1,
    M::T,
    B::S,
) where {T<:MpoTensor{R,4},S<:AbstractArray{R,3},S1<:AbstractArray{R,3}} where {R<:Real}
    for v ∈ M.top
        A = contract_tensor3_matrix(A, v)
    end
    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)
    end
    update_env_right(RE, A, M.ctr, B)
end

function update_env_right!(env::Environment, site::Site)
    site >= last(env.bra.sites) && return
    rs = right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs])
    ls = left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = contract_matrix_tensor3(env.mpo[ls], RR)
        ls = left_nbrs_site(ls, env.mpo.sites)
    end

    nRR = maximum(abs.(RR))
    RR ./= nRR
    push!(env.env, (site, :right) => RR)
    nRR = env.log_norms[(rs, :right)] + log(nRR)
    push!(env.log_norms, (site, :right) => nRR)
end

function update_env_right!(env::EnvironmentMixed{T}, site) where {T}   # site::Union(Sites, :central)
    if site == last(env.bra.sites)
        if env.site > last(env.bra.sites)
            RR = env.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
        else
            RR = env.onGPU ? CUDA.ones(T, 1, 1, 1) : ones(T, 1, 1, 1)
        end
    elseif site == :central
        rs = env.site
        RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs])
    elseif site >= env.site
        rs = right_nbrs_site(site, env.bra.sites)
        RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs])
        ls = left_nbrs_site(rs, env.mpo.sites)
        while ls > site
            RR = contract_matrix_tensor3(env.mpo[ls], RR)
            ls = left_nbrs_site(ls, env.mpo.sites)
        end
    else
        rs = right_nbrs_site(site, env.bra.sites)
        if rs == env.site
            RR = update_env_right(env.env[(:central, :right)], env.C)
        else
            RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.ket[rs])
        end
    end
    RR ./= maximum(abs.(RR))
    push!(env.env, (site, :right) => RR)
end


"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S,
    B::S,
    M::T,
    RE::S,
) where {S<:AbstractArray{R,3},T<:MpoTensor{R,4}} where {R<:Real}
    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)
    end
    B = project_ket_on_bra(LE, B, M.ctr, RE)
    for v ∈ reverse(M.top)
        B = contract_matrix_tensor3(v, B)
    end
    B
end


project_ket_on_bra(env::Environment, site::Site) = project_ket_on_bra(
    env.env[(site, :left)],
    env.ket[site],
    env.mpo[site],
    env.env[(site, :right)],
)

function project_ket_on_bra(env::EnvironmentMixed, site)
    if site == :central
        B = project_ket_on_bra(env.env[(site, :left)], env.env[(site, :right)])
    elseif site >= env.site
        B = project_ket_on_bra(
            env.env[(site, :left)],
            env.ket[site],
            env.mpo[site],
            env.env[(site, :right)],
        )
    else
        B = project_ket_on_bra(
            env.env[(site, :left)],
            env.ket[site],
            env.env[(site, :right)],
        )
    end
    B
end

function measure_env(env::Environment, site::Site)
    L = update_env_left(env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site])
    R = env.env[(site, :right)]
    overlap = @tensor L[b, t, c] * R[b, t, c]
    negative = overlap < 0
    overlap *= sign(overlap)
    (log(overlap) + env.log_norms[(site, :left)] + env.log_norms[(site, :right)], negative)
end
