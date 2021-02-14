#=
# This is the most general (still semi-sudo-code) of the search function.
# 
export low_energy_spectrum

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
     model::Model,
     sol::Solution,
    )
    boundary = zeros(Int, length(sol.states))
    for (i, σ) ∈ enumerate(sol.states)
        boundary[i] = generate_boundary(model, σ)
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

function _update_solution!(
    sol::Solution,
    model::Model, 
    )
    prob = eng = psol = [] 

    for (i, σ) ∈ enumerate(sol.states)
        p = conditional_probability(model, σ)
        push!(prob, sol.probabilities[i] .* p)
        push!(eng, sol.energies[i] .+ δE(model, σ))
        # add sol
    end

    sol = Solution(vec(eng), sol, vec(prob)) 
    _sort(_merge(model, sol), k)
end

@inline function _largest_discarded_probability!(
    lpCut::Float64,
    sol::Solution,
    )
    p = last(sol.probabilities)
    lpCut < p ? lpCut = p : ()
end

function low_energy_spectrum(
    model::Model, 
    k::Int
    )
    sol = Solution(zeros(k), fill([], k), zeros(k))
    lpCut = -typemax(Int)

    for v ∈ 1:model.size 
        _update_solution!(sol, model)
        _largest_discarded_probability!(lpCut, sol)
    end

    perm = partialsortperm(vec(sol.energies), 1:size(sol.energies), rev=true)
    sol = Solution(sol.energies[perm], sol.states[perm], sol.probabilities[perm])
    sol, lpCut
end
=#