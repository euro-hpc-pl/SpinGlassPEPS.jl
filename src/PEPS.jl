export PepsNetwork, contract_network
export MPO, MPS, generate_boundary, get_coordinates, update_energy

#TODO: use β from this setup  (currently β is passed directely to PepsNetwork)
function _set_control_parameters(
    args_override::Dict=Dict()
    )
    # put here more parameters if needs be
    args = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.,
        "β" => 1.
    )
    merge(args, args_override)
end

mutable struct PepsNetwork <: AbstractGibbsNetwork
    size::NTuple{2, Int}
    map::Dict
    network_graph::NetworkGraph
    origin::Symbol
    i_max::Int
    j_max::Int
    args::Dict

    function PepsNetwork(
        m::Int,
        n::Int,
        fg::MetaDiGraph,
        β::Number,
        origin::Symbol=:NW,
        args_override::Dict=Dict()
        )

        pn = new((m, n))
        pn.map, pn.i_max, pn.j_max = peps_indices(m, n, origin)

        nbrs = Dict()
        for i ∈ 1:pn.i_max, j ∈ 1:pn.j_max
            # v => (l, u, r, d)
            push!(nbrs,
            pn.map[i, j] => (pn.map[i, j-1], pn.map[i-1, j],
                             pn.map[i, j+1], pn.map[i+1, j]))
        end
        pn.network_graph = NetworkGraph(fg, nbrs, β)
        pn.args = _set_control_parameters(args_override)
        pn
    end
end

generate_tensor(pn::PepsNetwork,
                m::NTuple{2,Int},
                ) = generate_tensor(pn.network_graph, pn.map[m])

generate_tensor(pn::PepsNetwork,
                m::NTuple{2, Int},
                n::NTuple{2, Int},
                ) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])

generate_boundary(pn::PepsNetwork,
                  m::NTuple{2, Int},
                  n::NTuple{2, Int},
                  σ::Int,
                 ) = generate_boundary(pn.network_graph, pn.map[m], pn.map[n], σ)

function PEPSRow(::Type{T}, peps::PepsNetwork, i::Int) where {T <: Number}
    ψ = PEPSRow(T, peps.j_max)

    # generate tensors from projectors
    for j ∈ 1:length(ψ)
        ψ[j] = generate_tensor(peps, (i, j))
    end

    # include energy
    for j ∈ 1:peps.j_max
        A = ψ[j]
        h = generate_tensor(peps, (i, j-1), (i, j))
        v = generate_tensor(peps, (i-1, j), (i, j))
        @tensor B[l, u, r, d, σ] := h[l, l̃] * v[u, ũ] * A[l̃, ũ, r, d, σ]
        ψ[j] = B
    end
    ψ
end
PEPSRow(peps::PepsNetwork, i::Int) = PEPSRow(Float64, peps, i)

function MPO(::Type{T},
    peps::PepsNetwork,
    i::Int,
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) where {T <: Number}

    W = MPO(T, peps.j_max)
    R = PEPSRow(T, peps, i)

    for (j, A) ∈ enumerate(R)
        v = get(config, j + peps.j_max * (i - 1), nothing)
        if v !== nothing
            @cast B[l, u, r, d] |= A[l, u, r, d, $(v)]
        else
            @reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        end
        W[j] = B
    end
    W
end
MPO(peps::PepsNetwork,
    i::Int,
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) = MPO(Float64, peps, i, config)

function compress(ψ::AbstractMPS, peps::PepsNetwork)
    Dcut = peps.args["bond_dim"]
    if bond_dimension(ψ) < Dcut return ψ end
    compress(ψ, Dcut, peps.args["var_tol"], peps.args["sweeps"])
end

@memoize function MPS(
    peps::PepsNetwork,
    i::Int,
    cfg::Dict{Int, Int} = Dict{Int, Int}(),
    )
    if i > peps.i_max return MPS(I * 1.0) end
    W = MPO(peps, i, cfg)
    ψ = MPS(peps, i+1, cfg)
    compress(W * ψ, peps)
end

function contract_network(
    peps::PepsNetwork,
    config::Dict{Int, Int} = Dict{Int, Int}(),
    )
    ψ = MPS(peps, 1, config)
    prod(dropdims(ψ))[]
end

@inline function get_coordinates(
    peps::PepsNetwork,
    k::Int,
    )
    ceil(Int, k / peps.j_max), (k - 1) % peps.j_max + 1
end

@inline function _get_local_state(
    peps::PepsNetwork,
    v::Vector{Int},
    i::Int,
    j::Int,
    )
    k = j + peps.j_max * (i - 1)
    if k > length(v) || k <= 0 return 1 end
    v[k]
end

function generate_boundary(
    peps::PepsNetwork,
    v::Vector{Int},
    i::Int,
    j::Int,
    )
    ∂v = zeros(Int, peps.j_max + 1)

    # on the left below
    for k ∈ 1:j-1
        ∂v[k] = generate_boundary(peps,
            (i, k), (i+1, k),
            _get_local_state(peps, v, i, k))
    end

    # on the left at the current row
    ∂v[j] = generate_boundary(peps,
        (i, j-1), (i, j),
        _get_local_state(peps, v, i, j-1))

    # on the right above
    for k ∈ j:peps.j_max
        ∂v[k+1] = generate_boundary(peps,
            (i-1, k), (i, k),
            _get_local_state(peps, v, i-1, k))
    end
    ∂v
end

@inline function _contract(
    A::Array{T, 5},
    M::Array{T, 3},
    L::Vector{T},
    R::Matrix{T},
    ∂v::Vector{Int},
    ) where {T <: Number}

    l, u = ∂v
    @cast Ã[r, d, σ] := A[$l, $u, r, d, σ]
    @tensor prob[σ] := L[x] * M[x, d, y] *
                       Ã[r, d, σ] * R[y, r] order = (x, d, r, y)
    prob
end

function _normalize_probability(prob::Vector{T}) where {T <: Number}
    # exceptions (negative pdo, etc)
    # will be added here later
    prob / sum(prob)
end

function conditional_probability(
    peps::PepsNetwork,
    v::Vector{Int},
    )
    i, j = get_coordinates(peps, length(v)+1)
    ∂v = generate_boundary(peps, v, i, j)

    W = MPO(peps, i)
    ψ = MPS(peps, i+1)

    L = left_env(ψ, ∂v[1:j-1])
    R = right_env(ψ, W, ∂v[j+2:peps.j_max+1])
    A = generate_tensor(peps, (i, j))

    prob = _contract(A, ψ[j], L, R, ∂v[j:j+1])

    println("prob ", prob)
    ret = _normalize_probability(prob)
    
    #println("CP ", ret)
    ret
end

_bond_energy(pn::AbstractGibbsNetwork,
    ig::MetaGraph,
    m::NTuple{2, Int},
    n::NTuple{2, Int},
    σ::Int,
    ) = bond_energy(pn.network_graph, ig, pn.map[m], pn.map[n], σ)

_local_energy(pn::AbstractGibbsNetwork,
    m::NTuple{2, Int},
    ) = local_energy(pn.network_graph, pn.map[m])

function update_energy(
    network::AbstractGibbsNetwork,
    ig::MetaGraph,
    σ::Vector{Int},
    )
    i, j = get_coordinates(network, length(σ)+1)
    println("i, j ", i, j)

    σkj = _get_local_state(network, σ, i-1, j)
    σil = _get_local_state(network, σ, i, j-1)
    println("σil ", σil)
    println("σkj ", σkj)

    a = _bond_energy(network, ig, (i, j), (i, j-1), σil)
    b = _bond_energy(network, ig, (i, j), (i-1, j), σkj)
    c = _local_energy(network, (i, j))
    println("size a", size(a))
    println("a ", a)
    println("size b", size(b))
    println("b ", b)
    println("size c", size(c))
    println("c ", c)

    return c
end

function peps_indices(m::Int, n::Int, origin::Symbol=:NW)
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
