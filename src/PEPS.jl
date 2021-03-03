export PEPSNetwork, contract_network
export MPO, MPS, generate_boundary

const DEFAULT_CONTROL_PARAMS = Dict(
    "bond_dim" => typemax(Int),
    "var_tol" => 1E-8,
    "sweeps" => 4.,
    "β" => 1.
)

mutable struct PEPSNetwork
    size::NTuple{2, Int}
    map::Dict
    fg::MetaDiGraph
    nbrs::Dict
    origin::Symbol
    i_max::Int
    j_max::Int
    β::Number
    args::Dict{String, Number}

    function PEPSNetwork(
        m::Int,
        n::Int,
        fg::MetaDiGraph,
        β::Number,
        origin::Symbol=:NW,
        args_override::Dict{String, Number}=Dict{String, Number}()
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
        pn.fg = fg
        pn.nbrs = nbrs
        pn.β = β
        pn.args = merge(DEFAULT_CONTROL_PARAMS, args_override)
        pn
    end
end

generate_tensor(pn::PEPSNetwork,
                m::NTuple{2,Int},
                ) = generate_tensor(pn, pn.map[m])

generate_tensor(pn::PEPSNetwork,
                m::NTuple{2, Int},
                n::NTuple{2, Int},
                ) = generate_tensor(pn, pn.map[m], pn.map[n])

generate_boundary(pn::PEPSNetwork,
                  m::NTuple{2, Int},
                  n::NTuple{2, Int},
                  σ::Int,
                 ) = generate_boundary(pn, pn.map[m], pn.map[n], σ)

function PEPSRow(::Type{T}, peps::PEPSNetwork, i::Int) where {T <: Number}
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
PEPSRow(peps::PEPSNetwork, i::Int) = PEPSRow(Float64, peps, i)

function MPO(::Type{T},
    peps::PEPSNetwork,
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
MPO(peps::PEPSNetwork,
    i::Int,
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) = MPO(Float64, peps, i, config)

function compress(ψ::AbstractMPS, peps::PEPSNetwork)
    Dcut = peps.args["bond_dim"]
    if bond_dimension(ψ) < Dcut return ψ end
    compress(ψ, Dcut, peps.args["var_tol"], peps.args["sweeps"])
end

@memoize function MPS(
    peps::PEPSNetwork,
    i::Int,
    cfg::Dict{Int, Int} = Dict{Int, Int}(),
    )
    if i > peps.i_max return IdentityMPS() end
    W = MPO(peps, i, cfg)
    ψ = MPS(peps, i+1, cfg)
    compress(W * ψ, peps)
end

function contract_network(
    peps::PEPSNetwork,
    config::Dict{Int, Int} = Dict{Int, Int}(),
    )
    ψ = MPS(peps, 1, config)
    prod(dropdims(ψ))[]
end

@inline function _get_coordinates(
    peps::PEPSNetwork,
    k::Int
    )
    ceil(k / peps.j_max), (k - 1) % peps.j_max + 1
end

@inline function _get_local_state(
    peps::PEPSNetwork,
    v::Vector{Int},
    i::Int,
    j::Int,
    )
    k = j + peps.j_max * (i - 1)
    if k > length(v) || k <= 0 return 1 end
    v[k]
end

function generate_boundary(
    peps::PEPSNetwork,
    v::Vector{Int},
    i::Int,
    j::Int,
    )
    ∂v = zeros(Int, peps.j_max + 1)

    # on the left below
    for k ∈ 1:j-1
        ∂v[k] = generate_boundary(
            peps.network_graph,
            (i, k),
            (i+1, k),
            _get_local_state(peps, v, i, k))
    end

    # on the left at the current row
    ∂v[j] = generate_boundary(
        peps.network_graph,
        (i, j-1),
        (i, j),
        _get_local_state(peps, v, i, j-1))

    # on the right above
    for k ∈ j:peps.j_max
        ∂v[k+1] = generate_boundary(
            peps.network_graph,
            (i-1, k),
            (i, k),
            _get_local_state(peps, v, i-1, k))
    end
    ∂v
end

function generate_boundary(
    peps::PEPSNetwork,
    v::Vector{Int},
    )
    i, j = _get_coordinates(peps, length(v)+1)
    generate_boundary(peps, v, i, j)
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
    peps::PEPSNetwork,
    v::Vector{Int},
    )
    i, j = _get_coordinates(peps, length(v)+1)
    ∂v = generate_boundary(peps, v, i, j)

    W = MPO(peps, i)
    ψ = MPS(peps, i+1)

    L = left_env(ψ, ∂v[1:j-1])
    R = right_env(ψ, W, ∂v[j+2:peps.j_max+1])
    A = generate_tensor(peps, i, j)

    prob = _contract(A, ψ[j], L, R, ∂v[j:j+1])
    _normalize_probability(prob)
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
