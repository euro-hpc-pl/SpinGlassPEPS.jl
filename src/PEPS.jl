export PepsNetwork, contract
export MPO, MPS, boundaryMPS

function _set_control_parameters(
    args_override::Dict{String, Number}=Dict{String, Number}()
    )
    # put here more parameters if needs be
    args = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.,
        "β" => 1.
    )
    for k in keys(args_override)
        str = get(args_override, k, nothing)
        if str !== nothing push!(args, str) end
    end
    args
end

mutable struct PepsNetwork
    size::NTuple{2, Int}
    map::Dict
    network_graph::NetworkGraph
    origin::Symbol
    i_max::Int
    j_max::Int
    args::Dict{String, Number}

    function PepsNetwork(
        m::Int, 
        n::Int, 
        fg::MetaDiGraph,
        β::Number, 
        origin::Symbol=:NW,
        args_override::Dict{String, Number}=Dict{String, Number}()
        ) 

        pn = new((m, n))
        pn.map, pn.i_max, pn.j_max = LinearIndices(m, n, origin)

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
                m::NTuple{2,Int},
                n::NTuple{2,Int},
                ) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])

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
    if i > peps.i_max return MPS(I) end
    W, ψ = MPO(peps, i, cfg), MPS(peps, i+1, cfg)
    compress(W * ψ, peps)
end

function contract_network(
    peps::PepsNetwork,
    config::Dict{Int, Int} = Dict{Int, Int}(),
    ) 
    ψ = MPS(peps, 1, config)
    prod(dropdims(ψ))[]
end

#=
function conditional_probability(
    peps::PepsNetwork,
    v::Union{Vector{Int}, NTupel{Int}},
    )

    i = ceil(length(v) / peps.j_max)
    j = (length(v) - 1) % peps.j_max + 1

    ∂v = boundary(peps, v)

    ψ = MPS(peps, i+1)
    W = MPO(peps, i)

    L = left_env(ψ, ∂v[1:j-1])
    R = right_env(ψ, W, ∂v[j+2:peps.j_max+1])
    A = generate_tensor(peps, i, j)
 
    prob = contract(A, L, R, ∂v[j:j+1])
 
    normalize_prob(prob) 
end

function boundary(
    peps::PepsNetwork,
    v::Union(Vector{Int}, NTuple{Int}),
    )

    ∂v
end
=#