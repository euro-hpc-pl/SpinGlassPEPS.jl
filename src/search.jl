
# This is the most general (still semi-sudo-code) of the search function.
# 
export AbstractGibbsNetwork
export low_energy_spectrum

abstract type AbstractGibbsNetwork end

mutable struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
    largest_discarded_probability::Float64
end

#=
function _partition_into_unique(
    boundary::Vector{Int}, 
    partial_eng::Vector{T}
    ) where {T <: Number}
    
end

function _merge(
     network::AbstractGibbsNetwork,
     sol::Solution,
    )
    boundary = []
    for (i, σ) ∈ enumerate(sol.states)
        push!(boundary, generate_boundary(network, σ))
    end

    idx = _partition_into_unique(boundary, sol.energies)
    Solution(
        sol.energies[idx],
        sol.states[idx], 
        sol.probabilities[idx],
        sol.largest_discarded_probability)
end
=#

function _δE(
    ng::NetworkGraph, 
    v::Int, 
    w::Int, 
    σ::Vector{Int}
    )
    fg = ng.factor_graph
    loc_en = get_prop(fg, w, v, :loc_en)

    if has_edge(fg, w, v) 
        J = get_prop(fg, w, v, :edg).J
        return J
    elseif has_edge(fg, v, w)
        J = get_prop(fg, v, w, :edg).J
        return J
    else
        return nothing
    end
    energy(σ, J) + loc_en
end

_δE(pn::AbstractGibbsNetwork,
    m::NTuple{2, Int},
    n::NTuple{2, Int},
    σ::Vector{Int}
    ) = _δE(pn.network_graph, pn.map[m], pn.map[n], σ)

function _δE(
    network::AbstractGibbsNetwork,
    σ::Vector{Int}
    )
    i, j = get_coordinates(network, length(σ)+1)

    δE = 0

    # on the left below
    for k ∈ 1:j-1
        δE += δE(
            network.network_graph, 
            (i, k), 
            (i+1, k),
            σ)
    end

    # on the left at the current row
    δE = δE + δE(
        network.network_graph, 
        (i, j-1), 
        (i, j),
        σ)

    # on the right above
    for k ∈ j:peps.j_max
        δE += δE(
            network.network_graph,
            (i-1, k), 
            (i, k),
            σ)
    end
    δE
end

function _branch_and_bound(
    sol::Solution,
    network::AbstractGibbsNetwork, 
    node::Int,
    cut::Int,
    )
    # branch
    pdo = eng = cfg = []
    k = get_prop(network.factor_graph, node, :loc_dim)

    for (i, σ) ∈ enumerate(sol.states) 
        pdo = conditional_probability(network, σ)
        push!(pdo, (sol.probabilities[i] .* p)...)
        push!(eng, (sol.energies[i] .+ _δE(network, σ))...)
        push!(cfg, broadcast(s -> push!(sol.states[i], s), collect(1:k))...)
    end

    # bound
    idx = partialsortperm(pdo, 1:cut, rev=true)
    lpCut = sol.largest_discarded_probability 
    lpCut < last(pdo) ? lpCut = last(pdo) : ()
    Solution(eng[idx], cfg[idx], pdo[idx], lpCut)
end

function low_energy_spectrum(
    network::AbstractGibbsNetwork, 
    cut::Int
    )
    sol = Solution([0.], [[]], [1.], -Inf)
    for v ∈ 1:nv(network.factor_graph)
        sol = _branch_and_bound(sol, network, v, cut)
    end

    idx = partialsortperm(sol.energies, 1:length(sol.energies), rev=true)
    Solution(
        sol.energies[idx],
        sol.states[idx], 
        sol.probabilities[idx],
        sol.largest_discarded_probability)
end
