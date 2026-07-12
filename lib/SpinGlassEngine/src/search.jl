export SearchParameters,
    merge_branches,
    merge_branches_blur,
    low_energy_spectrum,
    Solution,
    bound_solution,
    gibbs_sampling,
    decode_to_spin,
    empty_solution,
    branch_energy,
    no_merge,
    sampling,
    branch_probability,
    discard_probabilities!,
    branch_energies,
    branch_states

"""
$(TYPEDSIGNATURES)
A struct representing search parameters for low-energy spectrum search.

## Constructor
Keyword arguments:
- `max_states::Int`: The maximum number of states to be considered during the search. Default is 1, indicating a single state search.
- `cutoff_prob::Real`: The cutoff probability for terminating the search. Default is 0.0, meaning no cutoff based on probability.

SearchParameters encapsulates parameters that control the behavior of low-energy spectrum search algorithms in the SpinGlassPEPS package.
"""
struct SearchParameters
    max_states::Int
    cutoff_prob::Real

    function SearchParameters(; max_states::Int = 1, cutoff_prob::Real = 0.0)
        new(max_states, cutoff_prob)
    end
end

"""
$(TYPEDSIGNATURES)
A struct representing a solution obtained from a low-energy spectrum search.

## Fields
- `energies::Vector{<:Real}`: A vector containing the energies of the discovered states.
- `states::Vector{Vector{Int}}`: A vector of state configurations corresponding to the energies.
- `probabilities::Vector{<:Real}`: The probabilities associated with each discovered state.
- `degeneracy::Vector{Int}`: The degeneracy of each energy level.
- `largest_discarded_probability::Real`: The largest probability of the largest discarded state.
- `droplets::Vector{Droplets}`: A vector of droplets associated with each state.
- `spins::Vector{Vector{Int}}`: The spin configurations corresponding to each state.

The `Solution` struct holds information about the results of a low-energy spectrum search, including the energy levels,
state configurations, probabilities, degeneracy, and additional details such as droplets and spin configurations.
Users can access this information to analyze and interpret the search results.
"""
struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
    droplets::Vector{Droplets}
    spins::Vector{Vector{Int}}
end

"""
$(TYPEDSIGNATURES)
Create an empty `Solution` object with a specified number of states.

This function creates an empty `Solution` object with the given number of states, initializing its fields with default values.

## Arguments
- `n::Int`: The number of states for which the `Solution` object is created.

## Returns
An empty `Solution` object with default field values, ready to store search results for a specified number of states.
"""
@inline empty_solution(::Type{T}, n::Int = 1) where {T} = Solution(
    zeros(T, n),
    fill(Vector{Int}[], n),
    zeros(T, n),
    ones(Int, n),
    T(-Inf),
    repeat([NoDroplets()], n),
    fill(Vector{Int}[], n),
)

"""
$(TYPEDSIGNATURES)
Create a new `Solution` object by selecting specific states from an existing `Solution`.

This constructor allows you to create a new `Solution` object by selecting specific states from an existing `Solution`.
It copies the energies, states, probabilities, degeneracy, droplets, and spins information for the selected states
while allowing you to set a custom `largest_discarded_probability`.

## Arguments
- `sol::Solution`: The original `Solution` object from which states are selected.
- `idx::Vector{Int}`: A vector of indices specifying the states to be selected from the original `Solution`.
- `ldp::Real=sol.largest_discarded_probability`: (Optional) The largest discarded probability for the new `Solution`.
By default, it is set to the largest discarded probability of the original `Solution`.

## Returns
A new `Solution` object containing information only for the selected states.
"""
function Solution(
    sol::Solution,
    idx::Vector{Int},
    ldp::Real = sol.largest_discarded_probability,
)
    Solution(
        sol.energies[idx],
        sol.states[idx],
        sol.probabilities[idx],
        sol.degeneracy[idx],
        ldp,
        sol.droplets[idx],
        sol.spins[idx],
    )
end

"""
$(TYPEDSIGNATURES)
Calculates the energy contribution of a branch given a base energy and a spin configuration.

This function calculates the energy contribution of a branch in a SpinGlassPEPS calculation.
It takes a `MpsContractor` object `ctr` and a tuple `eσ` containing a base energy as the first element
and a spin configuration represented as a vector of integers as the second element.
The function calculates the branch energy by adding the base energy to the energy contribution
of the given spin configuration obtained from the `update_energy` function.

## Arguments
- `ctr::MpsContractor{T}`: An instance of the `MpsContractor` type parameterized by the strategy type `T`.
- `eσ::Tuple{<:Real, Vector{Int}}`: A tuple containing the base energy as the first element (a real number)
and the spin configuration as the second element (a vector of integers).

## Returns
The branch energy, which is the sum of the base energy and the energy contribution of the spin configuration.
"""
@inline function branch_energy(
    ctr::MpsContractor{T},
    eσ::Tuple{<:Real,Vector{Int}},
) where {T}
    eσ[begin] .+ update_energy(ctr, eσ[end])
end

"""
$(TYPEDSIGNATURES)
Compute and branch the energies from different branches in a solution.

## Arguments
- `ctr::MpsContractor{T}`: The MPS contractor.
- `psol::Solution`: The partial solution.

## Returns
- `Vector{<:Real}`: A vector containing the energies of individual branches.

## Description
This function computes the energies of branches in a solution by applying the `branch_energy` function
to each pair of energy and state in the given partial solution.
The result is a vector of energies corresponding to the branches.
"""
@inline function branch_energies(ctr::MpsContractor{T}, psol::Solution) where {T}
    reduce(vcat, branch_energy.(Ref(ctr), zip(psol.energies, psol.states)))
end

"""
$(TYPEDSIGNATURES)
Constructs branch states based on a local basis and vectorized states.

## Arguments
- `local_basis::Vector{Int}`: The local basis states.
- `vec_states::Vector{Vector{Int}}`: Vectorized states for each branch.

## Returns
- `Vector{Vector{Int}}`: A vector containing the constructed branch states.

## Description
This function constructs branch states by combining a local basis with vectorized states.
The local basis provides the unique states for each branch, and the vectorized states represent the state configuration for each branch.
The resulting vector contains the constructed branch states.
"""
function branch_states(local_basis::Vector{Int}, vec_states::Vector{Vector{Int}})
    states = reduce(hcat, vec_states)
    num_states = length(local_basis)
    lstate, nstates = size(states)
    ns = Array{Int}(undef, lstate + 1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    collect(eachcol(reshape(ns, lstate + 1, nstates * num_states)))
end

"""
$(TYPEDSIGNATURES)
Calculates the branch probability for a given state.

## Arguments
- `ctr::MpsContractor{T}`: The MPS contractor object.
- `pσ::Tuple{<:Real, Vector{Int}}`: Tuple containing the energy and state configuration.

## Returns
- `Real`: The calculated branch probability.

## Description
This function calculates the branch probability for a specific state configuration using the conditional probability
provided by the MPS contractor.
The branch probability is computed as the logarithm of the conditional probability of the given state.
The conditional probability is obtained from the MPS contractor.
"""
function branch_probability(ctr::MpsContractor{T}, pσ::Tuple{<:Real,Vector{Int}}) where {T}
    pσ[begin] .+ log.(conditional_probability(ctr, pσ[end]))
end

"""
$(TYPEDSIGNATURES)
Discards low-probability states from the given solution.

## Arguments
- `psol::Solution`: The input solution containing states and their probabilities.
- `cutoff_prob::Real`: The cutoff probability below which states will be discarded.

## Returns
- `Solution`: A new solution with low-probability states discarded.

## Description
This function removes states from the solution `psol` whose probabilities are below the specified `cutoff_prob`.
It calculates a cutoff probability (`pcut`) based on the maximum probability in `psol` and the provided `cutoff_prob`.
States with probabilities lower than `pcut` are considered discarded.
The largest discarded probability (`ldp`) in the resulting solution is updated based on the
maximum discarded probability among the removed states and the existing `ldp` in `psol`.
"""
function discard_probabilities!(psol::Solution, cutoff_prob::Real)
    pcut = maximum(psol.probabilities) + log(cutoff_prob)
    if minimum(psol.probabilities) >= pcut
        return psol
    end
    local_ldp = maximum(psol.probabilities[psol.probabilities .< pcut])
    ldp = max(local_ldp, psol.largest_discarded_probability)
    Solution(psol, findall(p -> p >= pcut, psol.probabilities), ldp)
end

"""
$(TYPEDSIGNATURES)
Retrieve the local spin configurations associated with a vertex in the Gibbs network.

## Arguments
- `network::AbstractGibbsNetwork{S, T}`: The Gibbs network.
- `vertex::S`: The vertex for which local spins are requested.

## Returns
- `Vector{Int}`: An array representing the local spin configurations.

## Description
This function retrieves the local spin configurations associated with a given vertex in the Gibbs network.
The local spins are extracted from the spectrum of the Potts Hamiltonian associated with the vertex.
"""
function local_spins(network::AbstractGibbsNetwork{S,T}, vertex::S) where {S,T}
    spectrum(network, vertex).states_int
end

"""
$(TYPEDSIGNATURES)
Generate a new solution by branching the given partial solution in a contracting Gibbs network.

## Arguments
- `psol::Solution`: The partial solution.
- `ctr::T`: The contractor representing the contracting Gibbs network.

## Returns
- `Solution`: A new solution obtained by branching the partial solution in the contracting network.

## Description
This function generates a new solution by branching the given partial solution in a contracting Gibbs network.
It computes the energies, states, probabilities, degeneracies, discarded probabilities, droplets, and spins for the resulting solution.
The branching process involves considering the current node in the contractor and updating the solution accordingly.
"""
function branch_solution(psol::Solution, ctr::T) where {T<:AbstractContractor}
    num_states = cluster_size(ctr.peps, ctr.current_node)
    basis_states = collect(1:num_states)
    basis_spins = local_spins(ctr.peps, ctr.current_node)
    boundaries = boundary_states(ctr, psol.states, ctr.current_node)
    Solution(
        branch_energies(ctr, psol),
        branch_states(basis_states, psol.states),
        reduce(vcat, branch_probability.(Ref(ctr), zip(psol.probabilities, boundaries))),
        repeat(psol.degeneracy, inner = num_states),
        psol.largest_discarded_probability,
        repeat(psol.droplets, inner = num_states),#,
        branch_states(basis_spins, psol.spins),
    )
end

"""
$(TYPEDSIGNATURES)
Merge branches of a contractor based on specified merge type and droplet update strategy.

This function merges branches of a contractor (`ctr`) based on a specified merge type (`merge_prob`)
and an optional droplet update strategy (`droplets_encoding`).
It returns a function `_merge` that can be used to merge branches in a solution.

## Arguments
- `ctr::MpsContractor{T}`: A contractor for which branches will be merged.
- `merge_prob::Symbol=:none `: (Optional) The merge type to use. Defaults to `:none `. Possible values are `:none `, `:median`, and `:tnac4o`.
- `droplets_encoding=NoDroplets()`: (Optional) The droplet update strategy. Defaults to `NoDroplets()`. You can provide a custom droplet update strategy if needed.

## Returns
A function `_merge` that can be used to merge branches in a solution.

## Details
The `_merge` function can be applied to a `Solution` object to merge its branches based on the specified merge type and droplet update strategy.
"""
function merge_branches(
    ctr::MpsContractor{T};
    merge_prob::Symbol = :none,
    droplets_encoding = NoDroplets(),
) where {T}
    function _merge(psol::Solution)
        node = get(ctr.nodes_search_order, length(psol.states[1]) + 1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        _, bnd_types = SpinGlassNetworks.unique_dims(boundaries, 1)
        sorting_idx = sortperm(bnd_types)
        sorted_bnd_types = bnd_types[sorting_idx]
        nsol = Solution(psol, Vector{Int}(sorting_idx)) #TODO Vector{Int} should be rm
        energies = typeof(nsol.energies[begin])[]
        states = typeof(nsol.states[begin])[]
        spins = typeof(nsol.spins[begin])[]
        probs = typeof(nsol.probabilities[begin])[]
        degeneracy = typeof(nsol.degeneracy[begin])[]
        droplets = Droplets[]

        start = 1
        bsize = size(boundaries, 1)
        while start <= bsize
            stop = start
            while stop + 1 <= bsize && sorted_bnd_types[start] == sorted_bnd_types[stop+1]
                stop = stop + 1
            end
            best_idx_bnd = argmin(@view nsol.energies[start:stop])
            best_idx = best_idx_bnd + start - 1

            new_degeneracy = 0
            ind_deg = []
            for i = start:stop
                if nsol.energies[i] <= nsol.energies[best_idx] + 1E-12 # this is hack for now
                    new_degeneracy += nsol.degeneracy[i]
                    push!(ind_deg, i)
                end
            end

            if merge_prob == :median
                c = Statistics.median(
                    ctr.beta .* nsol.energies[start:stop] .+ nsol.probabilities[start:stop],
                )
                new_prob = -ctr.beta .* nsol.energies[best_idx] .+ c
                push!(probs, new_prob)
            elseif merge_prob == :none
                push!(probs, nsol.probabilities[best_idx])
            elseif merge_prob == :tnac4o
                push!(probs, Statistics.mean(nsol.probabilities[ind_deg]))
            end

            ## states with unique boundary => we take the one with best energy
            ## treat other states with the same boundary as droplets on top of the best one
            excitation = droplets_encoding(
                ctr,
                best_idx_bnd,
                nsol.energies[start:stop],
                nsol.states[start:stop],
                nsol.droplets[start:stop],
                nsol.spins[start:stop],
            )
            push!(droplets, excitation)

            push!(energies, nsol.energies[best_idx])
            push!(states, nsol.states[best_idx])
            push!(degeneracy, new_degeneracy)
            push!(spins, nsol.spins[best_idx])
            start = stop + 1
        end
        Solution(
            energies,
            states,
            probs,
            degeneracy,
            psol.largest_discarded_probability,
            droplets,
            spins,
        )
    end
    _merge
end

"""
$(TYPEDSIGNATURES)
Generate a function for merging branches in a Gibbs network with a Hamming distance blur.

## Arguments
- `ctr::MpsContractor{T}`: The contractor representing the contracted Gibbs network.
- `hamming_cutoff::Int`: The Hamming distance cutoff for blur.
- `merge_prob::Symbol=:none `: The merging strategy, defaults to `:none `.
- `droplets_encoding=NoDroplets()`: Droplet update method, defaults to `NoDroplets()`.

## Returns
A function `_merge_blur` that can be used to merge branches with Hamming distance blur in a solution.

## Description
This function generates a function for merging branches in a Gibbs network with Hamming distance blur.
The resulting function takes a partial solution as an input and performs the merging process, considering Hamming distance constraints.
It returns a new solution with the merged branches.
The Hamming distance blur helps in selecting diverse states during the merging process.
States with Hamming distances greater than or equal to the specified cutoff are considered distinct.
"""
function merge_branches_blur(
    ctr::MpsContractor{T},
    hamming_cutoff::Int,
    merge_prob::Symbol = :none,
    droplets_encoding = NoDroplets(),
) where {T}
    function _merge_blur(psol::Solution)
        psol = merge_branches(
            ctr;
            merge_prob = merge_prob,
            droplets_encoding = droplets_encoding,
        )(
            psol,
        )
        node = get(ctr.nodes_search_order, length(psol.states[1]) + 1, ctr.node_outside)
        boundaries = boundary_states(ctr, psol.states, node)
        sorted_indices = sortperm(psol.probabilities, rev = true)
        sorted_boundaries = boundaries[sorted_indices]
        nsol = Solution(psol, Vector{Int}(sorted_indices)) #TODO Vector{Int} should be rm
        selected_boundaries = []
        selected_idx = []
        for (i, state) in enumerate(sorted_boundaries)
            if all(
                hamming_distance(state, s, :Ising) >= hamming_cutoff for
                s in selected_boundaries
            ) #TODO case with :RMF
                push!(selected_boundaries, state)
                push!(selected_idx, i)
            end
        end
        Solution(nsol, Vector{Int}(selected_idx))
    end
    _merge_blur
end

"""
$(TYPEDSIGNATURES)
No-op merge function that returns the input `partial_sol` as is.

This function is a no-op merge function that takes a `Solution` object `partial_sol` as input and returns it unchanged.
It is used as a merge strategy when you do not want to perform any merging of branches in a solution.

## Arguments
- `partial_sol::Solution`: A `Solution` object representing partial solutions.

## Returns
The input `partial_sol` object, unchanged.
"""
no_merge(partial_sol::Solution) = partial_sol

"""
$(TYPEDSIGNATURES)
Bound the solution to a specified number of states while discarding low-probability states.

This function takes a `Solution` object `psol`, bounds it to a specified number of states `max_states`,
and discards low-probability states based on the probability threshold `δprob`.
You can specify a `merge_strategy` for merging branches in the `psol` object.

## Arguments
- `psol::Solution`: A `Solution` object representing the solution to be bounded.
- `max_states::Int`: The maximum number of states to retain in the bounded solution.
- `δprob::Real`: The probability threshold for discarding low-probability states.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.

## Returns
A `Solution` object representing the bounded solution with a maximum of `max_states` states.

"""
function bound_solution(
    psol::Solution,
    max_states::Int,
    δprob::Real,
    merge_strategy = no_merge,
)
    psol = discard_probabilities!(merge_strategy(psol), δprob)
    if length(psol.probabilities) <= max_states
        return psol
    end
    idx = partialsortperm(psol.probabilities, 1:max_states+1, rev = true)
    ldp = max(psol.largest_discarded_probability, psol.probabilities[idx[end]])
    Solution(psol, idx[1:max_states], ldp)
end

"""
$(TYPEDSIGNATURES)
Generate a new solution by sampling states based on their probabilities.

## Arguments
- `psol::Solution`: The partial solution from which to sample states.
- `max_states::Int`: The maximum number of states to sample.
- `δprob::Real`: The probability threshold for discarding states.
- `merge_strategy=no_merge`: The merging strategy, defaults to `no_merge`.

## Returns
- `Solution`: A new solution obtained by sampling states.

## Description
This function generates a new solution by sampling states from the given partial solution.
The sampling is performed based on the probabilities associated with each state.
The number of sampled states is determined by the `max_states` argument.
Additionally, states with probabilities below the threshold `δprob` are discarded.
The optional argument `merge_strategy` specifies the merging strategy to be used during the sampling process.
It defaults to `no_merge`, indicating no merging.
"""
function sampling(psol::Solution, max_states::Int, δprob::Real, merge_strategy = no_merge)
    prob = exp.(psol.probabilities)
    new_prob = cumsum(reshape(prob, :, max_states), dims = 1)

    rr = rand(max_states)
    idx = zeros(max_states)
    idx_lin = zeros(Int, max_states)
    for (i, m) in enumerate(rr)
        np = new_prob[:, i]
        new_prob[:, i] = np / np[end]
        idx[i] = searchsortedfirst(new_prob[:, i], m)
        idx_lin[i] = Int((i - 1) * size(new_prob, 1) + idx[i])
    end
    ldp = 0.0
    Solution(psol, idx_lin, ldp)
end

"""
$(TYPEDSIGNATURES)
Compute the low-energy spectrum on a quasi-2D graph using branch-and-bound search.

Merge matching configurations during branch-and-bound search going line by line.
Information about excited states (droplets) is collected during merging,
which allows reconstructing the low-energy spectrum.
It takes as input a `ctr` object representing the PEPS network and the parameters for controlling its contraction,
`sparams` specifying search parameters, `merge_strategy` for merging branches,
and `symmetry` indicating any symmetry constraints. Optionally, you can disable caching using the `no_cache` flag.
Probabilities are kept as log. Results are stored in Solution structure.

## Arguments
- `ctr::AbstractContractor`: The contractor object representing the PEPS network, which should be a subtype of `AbstractContractor`.
- `sparams::SearchParameters`: Parameters for controlling the search, including the maximum number of states and a cutoff probability.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.
- `symmetry::Symbol=:noZ2`: (Optional) Symmetry constraint. Defaults to `:noZ2`. If Z2 symmetry is present in your system, use `:Z2`.
- `no_cache=false`: (Optional) If `true`, disables caching. Defaults to `false`.

## Returns
A tuple `(sol, s)` containing:
- `sol::Solution`: A `Solution` object representing the computed low-energy spectrum.
- `s::Dict`: A dictionary containing Schmidt spectra for each row of the PEPS network.
"""
function low_energy_spectrum(
    ctr::MpsContractor{T,R,S},
    sparams::SearchParameters,
    merge_strategy = no_merge,
    symmetry::Symbol = :noZ2;
    no_cache = false,
) where {T,R,S}
    # Build all boundary mps
    CUDA.allowscalar(false)

    schmidts = Dict()
    @showprogress "Preprocessing: " for i ∈ ctr.peps.nrows+1:-1:2
        ψ0 = mps(ctr, i)
        push!(schmidts, i => measure_spectrum(ψ0))
        clear_memoize_cache_after_row()
        Memoization.empty_cache!(SpinGlassTensors.sparse)
        empty!(ctr.peps.lp, :GPU)
        if i <= ctr.peps.nrows
            ψ0 = mps(ctr, i + 1)
            move_to_CPU!(ψ0)
        end
    end

    s = Dict()
    for k in keys(schmidts)
        B = schmidts[k]
        v = []
        B = sort!(collect(B))
        for (i, _) in enumerate(B)
            push!(v, minimum(B[i][2]))
        end
        push!(s, k => v)
    end

    ψ0 = mps(ctr, 2)
    move_to_CPU!(ψ0)

    # Start branch and bound search
    sol = empty_solution(S)
    old_row = ctr.nodes_search_order[1][1]
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            clear_memoize_cache_after_row()
            empty!(ctr.peps.lp, :GPU)
        end
        sol = branch_solution(sol, ctr)
        if symmetry == :Z2 && length(sol.states[1]) == 1
            indices_with_even_numbers = Int[]
            for (index, vector) in enumerate(sol.spins)
                if any(iseven, vector)
                    push!(indices_with_even_numbers, index)
                end
            end
            # if !isempty(indices_with_odd_numbers)
            sol = Solution(sol, indices_with_even_numbers)
            # end
        end
        sol = bound_solution(sol, sparams.max_states, sparams.cutoff_prob, merge_strategy)
        Memoization.empty_cache!(precompute_conditional)
        if no_cache
            Memoization.empty_all_caches!()
        end
    end
    clear_memoize_cache_after_row()
    empty!(ctr.peps.lp, :GPU)

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.potts_hamiltonian.reverse_label_map[idx] for
        idx ∈ ctr.peps.vertex_map.(ctr.nodes_search_order)
    ])

    inner_perm_inv = zeros(Int, length(inner_perm))
    inner_perm_inv[inner_perm] = collect(1:length(inner_perm))

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability,
        [perm_droplet(drop, inner_perm_inv) for drop in sol.droplets[outer_perm]],
        sol.spins[outer_perm],
        # sol.pool_of_flips # TODO
    )

    # Final check if states correspond energies
    @assert sol.energies ≈
            energy.(
        Ref(ctr.peps.potts_hamiltonian),
        decode_state.(Ref(ctr.peps), sol.states),
    )
    sol, s
end

"""
$(TYPEDSIGNATURES)
Perform Gibbs sampling on a spin glass PEPS network.

This function performs Gibbs sampling on a spin glass PEPS (Projected Entangled Pair State) network using a branch-and-bound search algorithm. It takes as input a `ctr` object representing the PEPS network, `sparams` specifying search parameters, and `merge_strategy` for merging branches. Optionally, you can disable caching using the `no_cache` flag.

## Arguments

- `ctr::AbstractContractor`: The contractor object representing the PEPS network, which should be a subtype of `AbstractContractor`.
- `sparams::SearchParameters`: Parameters for controlling the search, including the maximum number of states and a cutoff probability.
- `merge_strategy=no_merge`: (Optional) Merge strategy for branches. Defaults to `no_merge`.
- `no_cache=false`: (Optional) If `true`, disables caching. Defaults to `false`.

## Returns

A `Solution` object representing the result of the Gibbs sampling.
"""
function gibbs_sampling(
    ctr::MpsContractor{T,R,S},
    sparams::SearchParameters,
    merge_strategy = no_merge;
    no_cache = false,
) where {T,R,S}
    # Build all boundary mps
    CUDA.allowscalar(false)

    @showprogress "Preprocessing: " for i ∈ ctr.peps.nrows:-1:1
        dressed_mps(ctr, i)
        clear_memoize_cache_after_row()
    end

    # Start branch and bound search
    sol = empty_solution(S, sparams.max_states)
    old_row = ctr.nodes_search_order[1][1]
    @showprogress "Search: " for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            clear_memoize_cache_after_row()
        end
        sol = branch_solution(sol, ctr)
        sol = sampling(sol, sparams.max_states, sparams.cutoff_prob, merge_strategy)
        Memoization.empty_cache!(precompute_conditional)
        # TODO: clear memoize cache partially
        if no_cache
            Memoization.empty_all_caches!()
        end
    end
    clear_memoize_cache_after_row()

    # Translate variable order (network --> factor graph)
    inner_perm = sortperm([
        ctr.peps.potts_hamiltonian.reverse_label_map[idx] for
        idx ∈ ctr.peps.vertex_map.(ctr.nodes_search_order)
    ])

    inner_perm_inv = zeros(Int, length(inner_perm))
    inner_perm_inv[inner_perm] = collect(1:length(inner_perm))

    # Sort using energies as keys
    outer_perm = sortperm(sol.energies)
    sol = Solution(
        sol.energies[outer_perm],
        [σ[inner_perm] for σ ∈ sol.states[outer_perm]],
        sol.probabilities[outer_perm],
        sol.degeneracy[outer_perm],
        sol.largest_discarded_probability,
        [perm_droplet(drop, inner_perm_inv) for drop in sol.droplets[outer_perm]],
        sol.spins[outer_perm],
    )

    # Final check if states correspond energies
    @assert sol.energies ≈
            energy.(
        Ref(ctr.peps.potts_hamiltonian),
        decode_state.(Ref(ctr.peps), sol.states),
    )
    sol
end
