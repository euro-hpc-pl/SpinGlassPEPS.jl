export SearchParameters,
    merge_branches,
    merge_branches_blur,
    low_energy_spectrum,
    Solution,
    bound_solution,
    gibbs_sampling,
    empty_solution,
    branch_energy,
    no_merge,
    sampling,
    branch_probability,
    discard_probabilities!,
    branch_energies,
    branch_states,
    verify_solution_energies!

# When true (default), low_energy_spectrum / gibbs_sampling verify at the end
# that every returned state decodes to its reported energy — a cheap-but-not-free
# self-check (O(num_states) energy re-evaluations, once per solve). Set
# `SpinGlassPEPS.verify_solution_energies!(false)` to skip it in throughput-
# bound production batch runs.
const VERIFY_SOLUTION_ENERGIES = Ref(true)
verify_solution_energies!(x::Bool) = (VERIFY_SOLUTION_ENERGIES[] = x)

"""
$(TYPEDSIGNATURES)
A struct representing search parameters for low-energy spectrum search.

## Constructor
Keyword arguments:
- `max_states::Int`: The maximum number of states to be considered during the search. Default is 1, indicating a single state search.
- `cutoff_prob::Real`: The cutoff probability for terminating the search. Default is 0.0, meaning no cutoff based on probability.
- `cut_off_prob::Real`: Deprecated spelling of `cutoff_prob`, retained for compatibility with published examples.

SearchParameters encapsulates parameters that control the behavior of low-energy spectrum search algorithms in the SpinGlassPEPS package.
"""
struct SearchParameters
    max_states::Int
    cutoff_prob::Real

    function SearchParameters(;
        max_states::Int = 1,
        cutoff_prob::Union{Real,Nothing} = nothing,
        cut_off_prob::Union{Real,Nothing} = nothing,
    )
        if cutoff_prob !== nothing && cut_off_prob !== nothing
            throw(
                ArgumentError(
                    "Specify only `cutoff_prob`; `cut_off_prob` is its deprecated alias.",
                ),
            )
        end
        if cut_off_prob !== nothing
            Base.depwarn(
                "SearchParameters(...; cut_off_prob=...) is deprecated, use cutoff_prob",
                :SearchParameters,
            )
        end
        resolved_cutoff = something(cutoff_prob, cut_off_prob, 0.0)
        new(max_states, resolved_cutoff)
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
    states::Vector{<:AbstractVector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
    droplets::Vector{Droplets}
    spins::Vector{<:AbstractVector{Int}}
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
    # `n` distinct empty configurations. This used to read `fill(Vector{Int}[], n)`,
    # which produced a `Vector{Vector{Vector{Int}}}` and only type-checked because
    # converting an *empty* `Vector{Vector{Int}}` to `Vector{Int}` happens to
    # succeed elementwise. Spelled out now that the field accepts any
    # `AbstractVector{Int}` element and no such conversion is available.
    [Int[] for _ = 1:n],
    zeros(T, n),
    ones(Int, n),
    T(-Inf),
    repeat([NoDroplets()], n),
    [Int[] for _ = 1:n],
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
    eσ::Tuple{<:Real,<:AbstractVector{Int}},
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
branch_states(local_basis::Vector{Int}, vec_states::AbstractVector{<:AbstractVector{Int}}) =
    [collect(σ) for σ ∈ branch_states_view(local_basis, vec_states)]

"""
$(TYPEDSIGNATURES)

Expansion used on the search's hot path: the same configurations
[`branch_states`](@ref) produces, but backed by one matrix and returned as column
views instead of independent vectors.

This is the largest single source of host-allocated bytes in a solve, previously
one heap object per branched state and so tens of thousands per call. The byte count
is dominated by the payload either way, but garbage-collection cost scales with the
*number* of objects, so consolidating the expansion into one matrix cuts the live
object count sharply for a modest change in bytes.

`branch_states` itself keeps returning `Vector{Vector{Int}}`, since it is part of
the published API and callers may rely on that element type.

Ordering is load-bearing: callers pair the result index-for-index with
`branch_energies` and `branch_probability`, so the local basis must vary fastest
within each parent configuration.
"""
function branch_states_view(
    local_basis::Vector{Int},
    vec_states::AbstractVector{<:AbstractVector{Int}},
)
    if isempty(vec_states)
        # Nothing to expand. Return an empty collection of the same element type
        # the non-degenerate branch produces, not a zero-column matrix view.
        empty = Matrix{Int}(undef, 0, 0)
        return [view(empty, :, c) for c = 1:0]
    end
    num_states = length(local_basis)
    nstates = length(vec_states)
    lstate = length(first(vec_states))
    M = Matrix{Int}(undef, lstate + 1, num_states * nstates)
    k = 0
    @inbounds for j = 1:nstates
        src = vec_states[j]
        for i = 1:num_states
            k += 1
            copyto!(view(M, 1:lstate, k), src)
            M[lstate+1, k] = local_basis[i]
        end
    end
    [view(M, :, c) for c = 1:size(M, 2)]
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
        branch_states_view(basis_states, psol.states),
        reduce(vcat, branch_probability.(Ref(ctr), zip(psol.probabilities, boundaries))),
        repeat(psol.degeneracy, inner = num_states),
        psol.largest_discarded_probability,
        repeat(psol.droplets, inner = num_states),#,
        branch_states_view(basis_spins, psol.spins),
    )
end

const _VALID_MERGE_PROBABILITIES = (:none, :median, :tnac4o)

function _canonical_merge_probability(
    merge_prob::Union{Symbol,Nothing},
    merge_type::Union{Symbol,Nothing},
)
    if merge_type !== nothing
        Base.depwarn(
            "merge_branches(...; merge_type=...) is deprecated, use merge_prob",
            :merge_branches,
        )
        legacy_merge_prob =
            merge_type ∈ _VALID_MERGE_PROBABILITIES ? merge_type :
            merge_type === :nofit ? :none :
            merge_type === :fit ? :median :
            merge_type === :python ? :tnac4o :
            throw(
                ArgumentError(
                    "Unknown merge type `$merge_type`; expected :none, :median, :tnac4o, :nofit, :fit, or :python.",
                ),
            )
        if merge_prob !== nothing && merge_prob !== legacy_merge_prob
            throw(
                ArgumentError(
                    "Conflicting merge strategies: merge_prob=$merge_prob and merge_type=$merge_type.",
                ),
            )
        end
        return legacy_merge_prob
    end

    canonical = something(merge_prob, :none)
    canonical ∈ _VALID_MERGE_PROBABILITIES || throw(
        ArgumentError(
            "Unknown merge probability strategy `$canonical`; expected one of $(_VALID_MERGE_PROBABILITIES).",
        ),
    )
    canonical
end

"""
    merge_branches(ctr; merge_prob=:none, droplets_encoding=NoDroplets(), ...)

Return a function that merges solution branches with the selected probability
strategy and optional droplet encoding. The deprecated `merge_type` and
`update_droplets` keywords remain supported for published 1.x examples.
"""
function merge_branches(
    ctr::MpsContractor{T};
    merge_prob::Union{Symbol,Nothing} = nothing,
    droplets_encoding = NoDroplets(),
    # deprecated names from the published API (SoftwareX 31, 102257); they
    # shipped in 1.x listings and must keep working until 2.0
    merge_type::Union{Symbol,Nothing} = nothing,
    update_droplets = nothing,
) where {T}
    merge_prob = _canonical_merge_probability(merge_prob, merge_type)
    if update_droplets !== nothing
        Base.depwarn(
            "merge_branches(...; update_droplets=...) is deprecated, use droplets_encoding",
            :merge_branches,
        )
        droplets_encoding = update_droplets
    end
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
- `show_progress=true`: (Optional) Display the preprocessing and search progress
  bars. Set to `false` when solving concurrently — see
  [`sweep_transformations`](@ref) — since interleaved bars are unreadable.
- `schmidt_spectrum=false`: (Optional) Compute the per-row Schmidt spectra
  returned as the second value. Off by default: it costs an untruncated CPU SVD
  per site per row and callers overwhelmingly discard the result. Note this is a
  change from 1.x, where the spectra were always computed.
- `retain_mps=false`: (Optional) Snapshot each row's bottom boundary MPS (on the
  host) into the contractor's warm-start slot as it is built, so that a following
  [`set_beta!`](@ref) can variationally compress from it rather than rebuilding
  from scratch. See [`beta_ladder`](@ref).

To record how much weight the boundary-MPS truncations discarded during the
solve, install a [`TruncationLog`](@ref) in the calling task's scope; see
[`sweep_transformations`](@ref), which does this per transformation.

## Returns
A tuple `(sol, s)` containing:
- `sol::Solution`: A `Solution` object representing the computed low-energy spectrum.
- `s::Dict`: A dictionary containing Schmidt spectra for each row of the PEPS
  network, empty unless `schmidt_spectrum=true`.
"""
function low_energy_spectrum(
    ctr::MpsContractor{T,R,S},
    sparams::SearchParameters,
    merge_strategy = no_merge,
    symmetry::Symbol = :noZ2;
    no_cache = false,
    show_progress::Bool = true,
    schmidt_spectrum::Bool = false,
    retain_mps::Bool = false,
) where {T,R,S}
    # Build all boundary mps
    CUDA.allowscalar(false)

    schmidts = Dict()
    prep = Progress(
        ctr.peps.nrows;
        desc = "Preprocessing: ",
        enabled = show_progress,
    )
    for i ∈ ctr.peps.nrows+1:-1:2
        ψ0 = mps(ctr, i)
        # An untruncated CPU SVD per site per row. Informative, but it is pure
        # overhead for the (overwhelmingly common) caller that discards the
        # second return value, so it is opt-in.
        schmidt_spectrum && push!(schmidts, i => measure_spectrum(ψ0))
        # Snapshot this row for a later β step, here rather than at the end of
        # the solve: `dressed_mps` takes ownership of each row's MPS as the
        # search absorbs it (`delete!(ctr.cache.mps, i + 1)`), so none of them
        # survive to the end. An independent host-side copy — sharing the object
        # would hand the next β a state the search has since moved to the device
        # and mutated. `mps` has already consumed any guess for this row, so
        # writing it back now cannot clobber one that is still needed.
        retain_mps && (ctr.guess[i] = move_to_CPU!(copy(ψ0)))
        probe_device_peak!()
        empty_row_caches!(ctr)
        empty!(ctr.peps.lp, :GPU)
        if i <= ctr.peps.nrows
            ψ0 = mps(ctr, i + 1)
            move_to_CPU!(ψ0)
        end
        next!(prep)
    end
    finish!(prep)

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
    search = Progress(
        length(ctr.nodes_search_order);
        desc = "Search: ",
        enabled = show_progress,
    )
    for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            probe_device_peak!()
            empty_row_caches!(ctr)
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
        empty!(ctr.cache.precond)
        if no_cache
            empty!(ctr.cache)
        end
        next!(search)
    end
    finish!(search)
    probe_device_peak!()
    empty_row_caches!(ctr)
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

    # Final check if states correspond energies (opt-out via VERIFY_SOLUTION_ENERGIES)
    VERIFY_SOLUTION_ENERGIES[] && @assert sol.energies ≈
            energy.(
        Ref(ctr.peps.potts_hamiltonian),
        decode_state.(Ref(ctr.peps), sol.states),
    )
    # The solve is done: release the contractor's caches now instead of
    # waiting for GC (test/benchmark loops build many contractors in a row,
    # and lingering device buffers fragment the CUDA pool).
    empty!(ctr.cache)
    empty!(ctr.peps.lp, :GPU)
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
- `show_progress=true`: (Optional) Display the preprocessing and search progress
  bars. Set to `false` when sampling concurrently, since interleaved bars are
  unreadable.

## Returns

A `Solution` object representing the result of the Gibbs sampling.
"""
function gibbs_sampling(
    ctr::MpsContractor{T,R,S},
    sparams::SearchParameters,
    merge_strategy = no_merge;
    no_cache = false,
    show_progress::Bool = true,
) where {T,R,S}
    # Build all boundary mps
    CUDA.allowscalar(false)

    prep =
        Progress(ctr.peps.nrows; desc = "Preprocessing: ", enabled = show_progress)
    for i ∈ ctr.peps.nrows:-1:1
        dressed_mps(ctr, i)
        probe_device_peak!()
        empty_row_caches!(ctr)
        next!(prep)
    end
    finish!(prep)

    # Start branch and bound search
    sol = empty_solution(S, sparams.max_states)
    old_row = ctr.nodes_search_order[1][1]
    search = Progress(
        length(ctr.nodes_search_order);
        desc = "Search: ",
        enabled = show_progress,
    )
    for node ∈ ctr.nodes_search_order
        ctr.current_node = node
        current_row = node[1]
        if current_row > old_row
            old_row = current_row
            probe_device_peak!()
            empty_row_caches!(ctr)
        end
        sol = branch_solution(sol, ctr)
        sol = sampling(sol, sparams.max_states, sparams.cutoff_prob, merge_strategy)
        empty!(ctr.cache.precond)
        # TODO: clear memoize cache partially
        if no_cache
            empty!(ctr.cache)
        end
        next!(search)
    end
    finish!(search)
    probe_device_peak!()
    empty_row_caches!(ctr)

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

    # Final check if states correspond energies (opt-out via VERIFY_SOLUTION_ENERGIES)
    VERIFY_SOLUTION_ENERGIES[] && @assert sol.energies ≈
            energy.(
        Ref(ctr.peps.potts_hamiltonian),
        decode_state.(Ref(ctr.peps), sol.states),
    )
    empty!(ctr.cache)
    empty!(ctr.peps.lp, :GPU)
    sol
end
