export tensor_assignment,
    tensor_species_map!,
    reduced_site_tensor,
    tensor_size,
    tensor,
    right_env,
    left_env,
    dressed_mps,
    mpo,
    mps

function assign_tensors!(network::PEPSNetwork)
    _types = (:site, :central_h, :central_v, :gauge_h)
    for type ∈ _types
        push!(network.tensor_spiecies, tensor_assignment(network, type)...)
    end
end


tensor_assignment(network::AbstractGibbsNetwork{S,T}, s::Symbol) where {S,T} =
    tensor_assignment(network, Val(s))


tensor_assignment(network::AbstractGibbsNetwork{S,T}, ::Val{:site}) where {S,T} =
    Dict((i, j) => :site for i ∈ 1:network.nrows, j ∈ 1:network.ncols)


tensor_assignment(network::AbstractGibbsNetwork{S,T}, ::Val{:central_h}) where {S,T} =
    Dict((i, j + 1 // 2) => :central_h for i ∈ 1:network.nrows, j ∈ 1:network.ncols)


tensor_assignment(network::AbstractGibbsNetwork{S,T}, ::Val{:central_v}) where {S,T} =
    Dict((i + 1 // 2, j) => :central_v for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols)


tensor_assignment(network::PEPSNetwork, ::Val{:gauge_h}) = Dict(
    (i + δ, j) => :gauge_h for i ∈ 1:network.nrows-1, j ∈ 1:network.ncols,
    δ ∈ (1 // 6, 2 // 6, 4 // 6, 5 // 6)
)


function tensor(network::AbstractGibbsNetwork, v)
    if v ∈ keys(network.tensor_spiecies)
        tensor(network, v, Val(network.tensor_spiecies[v]))
    else
        ones(1, 1, 1, 1)
    end
end


function tensor_size(network::AbstractGibbsNetwork, v)
    if v ∈ keys(network.tensor_spiecies)
        tensor_size(network, v, Val(network.tensor_spiecies[v]))
    else
        (1, 1, 1, 1)
    end
end


function tensor(network::AbstractGibbsNetwork{S}, v::S, ::Val{:site}) where {S}
    loc_exp = exp.(-network.β .* local_energy(network, v))
    projs = projectors(network, v)
    # @cast A[σ, _] := loc_exp[σ]
    A = reshape(loc_exp, :, 1)
    for pv ∈ projs
        # @cast A[σ, (c, γ)] |= A[σ, c] * pv[σ, γ]
        A .= reshape(z .* reshape(pv, size(pv, 1), 1, size(pv, 2)), size(z, 1), :)
    end
    B = dropdims(sum(A, dims = 1), dims = 1)
    reshape(B, size.(projs, 2))
end


function tensor_size(network::AbstractGibbsNetwork, v, ::Val{:site})
    dims = size.(projectors(network, v))
    pdims = first.(dims)
    @assert all(σ -> σ == first(pdims), pdims)
    last.(dims) # TODO: CZEMU?
end


function tensor(
    network::AbstractGibbsNetwork,
    v::Tuple{Rational{Int},Int},
    ::Val{:central_v},
)
    r, j = v
    i = floor(Int, r)
    h = connecting_tensor(network, (i, j), (i + 1, j))
    # @cast A[_, u, _, d] := h[u, d]
    A = reshape(h, 1, size(h, 1), 1, size(h, 2))
    A
end


function tensor_size(
    network::AbstractGibbsNetwork,
    v::Tuple{Rational{Int},Int},
    ::Val{:central_v},
)
    r, j = v
    i = floor(Int, r)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j)))
    (1, u, 1, d)
end


function tensor(
    network::AbstractGibbsNetwork,
    w::Tuple{Int,Rational{Int}},
    ::Val{:central_h},
)
    i, r = w
    j = floor(Int, r)
    v = connecting_tensor(network, (i, j), (i, j + 1))
    # @cast A[l, _, r, _] := v[l, r]
    A = reshape(v, size(v, 1), 1, size(v, 2), 1)
    A
end


function tensor_size(
    network::AbstractGibbsNetwork,
    w::Tuple{Int,Rational{Int}},
    ::Val{:central_h},
)
    i, r = w
    j = floor(Int, r)
    l, r = size(interaction_energy(network, (i, j), (i, j + 1)))
    (l, 1, r, 1)
end


function tensor(
    network::AbstractGibbsNetwork,
    v::Tuple{Rational{Int},Rational{Int}},
    ::Val{:central_d},
)
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    NW = connecting_tensor(network, (i, j), (i + 1, j + 1))
    NE = connecting_tensor(network, (i, j + 1), (i + 1, j))
    # @cast A[_, (u, ũ), _, (d, d̃)] := NW[u, d] * NE[ũ, d̃]
    u, d = size(NW)
    ũ, d̃ = size(NE)
    A = reshape(reshape(NW, u, 1, d) .* reshape(NE, 1, ũ, 1, d̃), 1, u * ũ, 1, d * d̃)
    A
end


function tensor_size(
    network::AbstractGibbsNetwork,
    v::Tuple{Rational{Int},Rational{Int}},
    ::Val{:central_d},
)
    r, s = v
    i = floor(Int, r)
    j = floor(Int, s)
    u, d = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    ũ, d̃ = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (1, u * ũ, 1, d * d̃)
end


function _all_fused_projectors(network::AbstractGibbsNetwork, v::Tuple{Int,Rational{Int}})
    i, s = v
    j = floor(Int, s)

    left_nbrs = ((i + 1, j + 1), (i, j + 1), (i - 1, j + 1))
    prl = projector.(Ref(network), Ref((i, j)), left_nbrs)
    p_lb, p_l, p_lt = last(fuse_projectors(prl))

    right_nbrs = ((i + 1, j), (i, j), (i - 1, j))
    prr = projector.(Ref(network), Ref((i, j + 1)), right_nbrs)
    p_rb, p_r, p_rt = last(fuse_projectors(prr))

    p_lb, p_l, p_lt, p_rb, p_r, p_rt
end


function tensor(network::AbstractGibbsNetwork, v::Tuple{Int,Rational{Int}}, ::Val{:virtual})
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = _all_fused_projectors(network, v)

    h = connecting_tensor(network, floor.(Int, v), ceil.(Int, v))

    @tensor B[l, r] := p_l[l, x] * h[x, y] * p_r[r, y]
    # @cast A[l, (ũ, u), r, (d̃, d)] |=
    #     B[l, r] * p_lt[l, u] * p_rb[r, d] * p_rt[r, ũ] * p_lb[l, d̃]
    ((l, r), u, d, ũ, d̃) =
        size(B), size(p_lt, 2), size(p_rb, 2), size(p_rt, 2), size(p_lb, 2)
    A = reshape(
        reshape(B, l, 1, 1, r) .* reshape(p_lt, l, 1, u) .*
        reshape(p_rb, 1, 1, 1, r, 1, d) .* reshape(p_rt', 1, ũ, 1, r) .*
        reshape(p_lb, l, 1, 1, 1, d̃),
        l,
        ũ * u,
        r,
        d̃ * d,
    )
    A
end


function tensor_size(
    network::AbstractGibbsNetwork,
    v::Tuple{Int,Rational{Int}},
    ::Val{:virtual},
)
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = _all_fused_projectors(network, v)
    (
        size(p_l, 1),
        size(p_lt, 2) * size(p_rt, 2),
        size(p_r, 1),
        size(p_rb, 2) * size(p_lb, 2),
    )
end


function tensor(network::AbstractGibbsNetwork, v, ::Val{:gauge_h})
    X = network.gauges[v]
    # @cast A[_, u, _, d] := Diagonal(X)[u, d]
    A = reshape(Diagonal(X), 1, length(X), 1, length(X))
    A
end


function tensor_size(network::AbstractGibbsNetwork, v, ::Val{:gauge_h})
    X = network.gauges[v]
    u = size(X, 1)
    (1, u, 1, u)
end


function connecting_tensor(network::AbstractGibbsNetwork{S}, v::S, w::S) where {S}
    en = interaction_energy(network, v, w)
    exp.(-network.β .* (en .- minimum(en)))
end


function reduced_site_tensor(network::PEPSNetwork, v::Tuple{Int,Int}, l::Int, u::Int)
    i, j = v
    eng_local = local_energy(network, v)
    pl = projector(network, v, (i, j - 1))
    eng_pl = interaction_energy(network, v, (i, j - 1))
    # @matmul eng_left[x] := sum(y) pl[x, y] * eng_pl[y, $l]
    @tensor eng_left[x] := pl[x, y] * view(eng_pl, :, l)[y]

    pu = projector(network, v, (i - 1, j))
    eng_pu = interaction_energy(network, v, (i - 1, j))
    # @matmul eng_up[x] := sum(y) pu[x, y] * eng_pu[y, $u]
    @tensor eng_up[x] := pu[x, y] * view(eng_pu, :, u)[y]

    en = eng_local .+ eng_left .+ eng_up
    loc_exp = exp.(-network.β .* (en .- minimum(en)))

    pr = projector(network, v, (i, j + 1))
    pd = projector(network, v, (i + 1, j))
    # @cast A[r, d, σ] := pr[σ, r] * pd[σ, d] * loc_exp[σ]
    A =
        reshape(pr', size(pr, 2), 1, size(pr, 1)) .*
        reshape(pd', 1, size(pd, 2), size(pd, 1)) .* reshape(loc_exp, 1, 1, :)
    A
end


function tensor_size(network::PEPSNetwork, v::Tuple{Int,Int}, ::Val{:reduced})
    i, j = v
    pr = projector(network, v, (i, j + 1))
    pd = projector(network, v, (i + 1, j))
    @assert size(pr, 1) == size(pr, 1)
    (size(pr, 2), size(pd, 2), size(pd, 1))
end


function mpo(
    ::Type{T},
    peps::AbstractGibbsNetwork,
    r::Union{Rational{Int},Int},
) where {T<:Number}
    W = MPO(T, length(peps.columns_MPO) * peps.ncols)
    layers = Iterators.product(peps.columns_MPO, 1:peps.ncols)
    Threads.@threads for (k, (d, j)) ∈ collect(enumerate(layers))
        W[k] = tensor(peps, (r, j + d))
    end
    W
end


@memoize Dict mpo(
    peps::AbstractGibbsNetwork{T,S,R},
    r::Union{Rational{Int},Int},
) where {T,S,R} = mpo(R, peps, r)



function mps(::Type{T}, peps::AbstractGibbsNetwork, i::Int) where {T<:Number}
    if i > peps.nrows
        return IdentityMPS()
    end
    ψ = mps(peps, i + 1)
    for r ∈ peps.layers_MPS
        ψ = mpo(peps, i + r) * ψ
    end
    compress(ψ, peps)
end


@memoize Dict mps(peps::AbstractGibbsNetwork{T,S,R}, i::Int) where {T,S,R} = mps(R, peps, i)


@memoize Dict function dressed_mps(peps::AbstractGibbsNetwork, i::Int)
    ψ = mps(peps, i + 1)
    for r ∈ peps.layers_left_env
        ψ = mpo(peps, i + r) * ψ
    end
    ψ
end


function compress(ψ::AbstractMPS, peps::AbstractGibbsNetwork)
    if bond_dimension(ψ) < peps.bond_dim
        return ψ
    end
    SpinGlassTensors.compress(ψ, peps.bond_dim, peps.var_tol, peps.sweeps)
end


@memoize Dict function _mpo(peps::AbstractGibbsNetwork, i::Int)
    prod(mpo.(Ref(peps), i .+ reverse(peps.layers_right_env)))
end


@memoize Dict function right_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    l = length(∂v)
    if l == 0
        return ones(1, 1)
    end
    R̃ = right_env(peps, i, ∂v[2:l])
    ϕ = dressed_mps(peps, i)
    W = _mpo(peps, i)
    k = length(W)
    M = ϕ[k-l+1]
    M̃ = W[k-l+1]
    K = @view M̃[:, ∂v[1], :, :]
    @tensor R[x, y] := K[y, β, γ] * M[x, γ, α] * R̃[α, β] order = (β, γ, α)
    R
end


@memoize Dict function left_env(peps::AbstractGibbsNetwork, i::Int, ∂v::Vector{Int})
    l = length(∂v)
    if l == 0
        return ones(1)
    end
    L̃ = left_env(peps, i, ∂v[1:l-1])
    ϕ = dressed_mps(peps, i)
    m = ∂v[l]
    M = ϕ[l]
    # @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    @tensor L[x] := L̃[α] * view(M, :, m, :)[α, x]
    L
end
