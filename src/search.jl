
# This is the most general (still semi-sudo-code) of the search function.
# 
export AbstractGibbsNetwork
export low_energy_spectrum

abstract type AbstractGibbsNetwork end

mutable struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
end

function _partition_into_unique(
    boundary::Vector{Int}, 
    partial_eng::Vector{T}
    ) where {T <: Number}

end

function _merge(
     network::AbstractGibbsNetwork,
     sol::Solution,
    )
    boundary = zeros(Int, length(sol.states))
    for (i, σ) ∈ enumerate(sol.states)
        boundary[i] = generate_boundary(network, σ)
    end

    idx = _partition_into_unique(boundary, sol.energies)
    Solution(sol.energies[idx], sol.states[idx], sol.probabilities[idx])
end

function _sort(
    sol::Solution,
    k::Int,
    )
    perm = partialsortperm(sol.probabilities, 1:k, rev=true)
    Solution(sol.energies[perm], sol.states[perm], sol.probabilities[perm])
end

function _δE(
    network::AbstractGibbsNetwork
    σ::Vector{Vector{Int}}
)
end
#=
function _branch_state(network::AbstractGibbsNetwork, state::Vec, i::Int)
    k = get_prop(network.factor_graph, i, :loc_dim)
    vcat(repeat(state, 1, length(k)), reshape(k, 1, :))
end
=#

function _branch_and_bound!(
    solution::Solution,
    network::AbstractGibbsNetwork, 
    k::Int,
    )
    # branch
    new = Solution([], [[]], [])
    for (i, σ) ∈ enumerate(solution.states) 
        pdo = conditional_probability(network, σ)

        push!(new.probabilities, solution.probabilities[i] .* p)
        push!(new.energies, solution.energies[i] .+ _δE(network, σ))
        #_branch_state(network, σ, i)
        k = get_prop(network.factor_graph, i, :loc_dim)
        broadcast(s -> push!(solution.states[i], s), collect(1:k))
    end

    # bound
    sol = Solution(vec(solution.energies), vec(solution.states), vec(solution.probabilities))
    _sort(_merge(network, sol), k)
end

@inline function _largest_discarded_probability!(
    lpCut::Float64,
    sol::Solution,
    )
    p = last(sol.probabilities)
    lpCut < p ? lpCut = p : ()
end

function low_energy_spectrum(
    network::AbstractGibbsNetwork, 
    k::Int
    )
    sol = Solution(zeros(k), fill([], k), zeros(k))
    lpCut = -typemax(Int)

    for v ∈ 1:network.size 
        _branch_and_bound!(sol, network, k)
        _largest_discarded_probability!(lpCut, sol)
    end

    perm = partialsortperm(vec(sol.energies), 1:size(sol.energies), rev=true)
    sol = Solution(sol.energies[perm], sol.states[perm], sol.probabilities[perm])
    sol, lpCut
end
