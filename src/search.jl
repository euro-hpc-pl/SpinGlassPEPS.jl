export AbstractGibbsNetwork
export low_energy_spectrum
export Solution

abstract type AbstractGibbsNetwork end

mutable struct Solution
    #energies::Vector{Float64}
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

    # for (p, σ, e) ∈ zip(sol.probabilities, sol.states, sol.energies)
    #     pdo = [pdo; p .* conditional_probability(network, σ)]
    #     eng = [eng; e .+ update_energy(network, σ)]
    #     cfg = _branch_state(cfg, σ, collect(1:k))
    # end

    for (p, σ) ∈ zip(sol.probabilities, sol.states)
        println("sigma ", σ)
        pdo = [pdo; p .* conditional_probability(network, σ)]
        cfg = _branch_state(cfg, σ, collect(1:k))
    end

    # bound
    idx = partialsortperm(pdo, 1:min(length(pdo), cut), rev=true)
    lpCut = sol.largest_discarded_probability
    lpCut < last(pdo) ? lpCut = last(pdo) : ()

    # Solution(eng[idx], cfg[idx], pdo[idx], lpCut)
    Solution(cfg[idx], pdo[idx], lpCut)
end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    network::AbstractGibbsNetwork,
    cut::Int
    )
    ng = network.network_graph

    #sol = Solution([0.], [[]], [1.], -Inf)
    sol = Solution([[]], [1.], -Inf)
    for v ∈ 1:nv(ng.factor_graph)
        sol = _branch_and_bound(sol, network, v, cut)
    end

    #idx = partialsortperm(sol.energies, 1:length(sol.energies), rev=true)
    Solution(
        # sol.energies[idx],
        # sol.states[idx],
        # sol.probabilities[idx],
        sol.states,
        sol.probabilities,
        sol.largest_discarded_probability)
end
