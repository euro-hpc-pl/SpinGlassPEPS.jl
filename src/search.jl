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

# function _partition_into_unique(
#     boundary::Vector{Int},
#     partial_eng::Vector{T}
#     ) where {T <: Number}

# end

# function _merge(
#      network::AbstractGibbsNetwork,
#      sol::Solution,
#     )
#     boundary = []
#     for (i, σ) ∈ enumerate(sol.states)
#         push!(boundary, generate_boundary(network, σ))
#     end

#     idx = _partition_into_unique(boundary, sol.energies)
#     Solution(
#         sol.energies[idx],
#         sol.states[idx],
#         sol.probabilities[idx],
#         sol.largest_discarded_probability)
# end

#TODO: this can probably be done better
function _branch_state(
    cfg::Vector,
    state::Vector,
    basis::Vector,
    )
    tmp = Vector{Int}[]
    for σ ∈ basis push!(tmp, vcat(state, σ)) end
    vcat(cfg, tmp)
end

@inline _init_solution() = (Float64[], Float64[], Vector{Int}[])

# TODO: logic here can probably be done better
function _bound(pdo::Vector{Float64}, cut::Int)
    k = length(pdo)
    second_phase = false
    if k > cut + 1 k = cut + 1; second_phase = true end
   
    idx = partialsortperm(pdo, 1:k, rev=true)

    if second_phase  
        return idx[1:end-1], pdo[last(idx)]
    else
        return idx, -Inf
    end
end

function _branch_and_bound(
    sol::Solution,
    network::AbstractGibbsNetwork,
    node::Int,
    cut::Int,
    )
    ng = network.network_graph
    fg = ng.factor_graph

    # branch
    pdo, eng, cfg = _init_solution()
    k = get_prop(fg, node, :loc_dim)

    for (p, σ, e) ∈ zip(sol.probabilities, sol.states, sol.energies)
        pdo = [pdo; p .* conditional_probability(network, σ)]
        eng = [eng; e .+ update_energy(network, σ)]
        cfg = _branch_state(cfg, σ, collect(1:k))
     end

    # bound
    K, lp = _bound(pdo, cut)
    lpCut = sol.largest_discarded_probability
    lpCut < lp ? lpCut = lp : () 

    Solution(eng[K], cfg[K], pdo[K], lpCut)
end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    network::AbstractGibbsNetwork,
    cut::Int
    )
    ng = network.network_graph

    sol = Solution([0.], [[]], [1.], -Inf)
    for v ∈ 1:nv(ng.factor_graph)
        sol = _branch_and_bound(sol, network, v, cut)
    end

    K = partialsortperm(sol.energies, 1:length(sol.energies), rev=false)
    Solution(
        sol.energies[K],
        sol.states[K],
        sol.probabilities[K],
        sol.largest_discarded_probability)
end

