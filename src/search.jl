
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
    new([0.], [[]], [1.], -Inf)
end


function _partition_into_unique(
    boundary::Vector{Int}, 
    partial_eng::Vector{T}
    ) where {T <: Number}
    
end

#=
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
    network::AbstractGibbsNetwork
    σ::Vector{Int}
    )

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

    for (i, σ) ∈ enumerate(solution.states) 
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
    sol = Solution()
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
