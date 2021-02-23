export AbstractGibbsNetwork
export low_energy_spectrum
export Solution

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

function _branch_and_bound(
    sol::Solution,
    network::AbstractGibbsNetwork, 
    node::Int,
    cut::Int,
    )
    ng = network.network_graph
    fg = ng.factor_graph

    # branch
    pdo = eng = cfg = []
    k = get_prop(fg, node, :loc_dim)

    for (i, σ) ∈ enumerate(sol.states) 
        pdo = conditional_probability(network, σ)
        push!(pdo, (sol.probabilities[i] .* pdo)...)
        push!(eng, (sol.energies[i] .+ update_energy(network, σ))...)
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
    ng = network.network_graph

    sol = Solution([0.], [[]], [1.], -Inf)
    for v ∈ 1:nv(ng.factor_graph)
        sol = _branch_and_bound(sol, network, v, cut)
        #TODO: incorportae "going back" move to improve alghoritm 
    end

    idx = partialsortperm(sol.energies, 1:length(sol.energies), rev=true)
    Solution(
        sol.energies[idx],
        sol.states[idx], 
        sol.probabilities[idx],
        sol.largest_discarded_probability)
end
