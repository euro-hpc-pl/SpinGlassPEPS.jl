# droplets.jl: This file provides functions for finding droplets and operating on them.

export NoDroplets,
    SingleLayerDroplets,
    Flip,
    Droplet,
    Droplets,
    hamming_distance,
    unpack_droplets,
    perm_droplet,
    filter_droplets,
    my_push!,
    diversity_metric,
    merge_droplets,
    flip_state

struct NoDroplets end

Base.iterate(drop::NoDroplets) = nothing
Base.copy(s::NoDroplets) = s
Base.getindex(s::NoDroplets, ::Any) = NoDroplets()

"""
A data structure representing the properties and criteria for identifying single-layer droplets in the context of the SpinGlassPEPS package.

A `SingleLayerDroplets` object is used to specify the maximum energy, minimum size, and metric for single-layer droplets in the SpinGlassPEPS.

## Fields
- `max_energy::Real`: The maximum allowed excitation energy above the ground state. It is typically a real number.
- `min_size::Int`: The minimum Hamming distance required between excitations for them to be considered distinct. 
- `metric::Symbol`: The metric used to evaluate the significance of a single-layer droplet. Default is `:no_metric`. `:hamming` treats Hamming distances as a metric.
- `mode::Symbol`: `:Ising` assumes Ising-type representation of the problem. `:RMF` assumes a Random Markov Field type model. Default is `:Ising`.

## Constructors
- `SingleLayerDroplets(max_energy::Real = 1.0, min_size::Int = 1, metric::Symbol = :no_metric, mode = :Ising)`: Creates a new `SingleLayerDroplets` object with the specified maximum energy, minimum size, metric and mode.

"""
struct SingleLayerDroplets
    max_energy::Real
    min_size::Int
    metric::Symbol
    mode::Symbol
    SingleLayerDroplets(;
        max_energy = 1.0,
        min_size = 1,
        metric = :no_metric,
        mode = :Ising,
    ) = new(max_energy, min_size, metric, mode)
end

"""
A data structure representing a set of flips or changes in states for nodes in the SpinGlassPEPS package.

A `Flip` object contains information about the support, state changes, and spinxor values for a set of node flips in the SpinGlassPEPS system.

## Fields
- `support::Vector{Int}`: An array of integers representing the indices of nodes where flips occur.
- `state::Vector{Int}`: An array of integers representing the new states for the nodes in the `support`.
- `spinxor::Vector{Int}`: An array of integers representing the spin-xor values for the nodes in the `support`.

## Constructors
- `Flip(support::Vector{Int}, state::Vector{Int}, spinxor::Vector{Int})`:
Creates a new `Flip` object with the specified support, state changes, and spinxor values.

"""
struct Flip
    support::Vector{Int}
    state::Vector{Int}
    spinxor::Vector{Int}
    statexor::Vector{Int}
end

"""
$(TYPEDSIGNATURES)
A data structure representing a droplet in the context of the SpinGlassPEPS package.
A `Droplet` represents an excitation in the SpinGlassPEPS system. It contains information about the excitation energy,
the site where the droplet starts, the site where it ends, the states of nodes flipped by the droplet,
and any sub-droplets on top of the current droplet.

## Fields
- `denergy::Real`: The excitation energy of the droplet, typically a real number.
- `first::Int`: The site index where the droplet starts.
- `last::Int`: The site index where the droplet ends.
- `flip::Flip`: The states of nodes flipped by the droplet, often represented using a `Flip` type.
- `droplets::Union{NoDroplets, Vector{Droplet}}`: A field that can be either `NoDroplets()` if there are no sub-droplets
on top of the current droplet or a vector of `Droplet` objects representing sub-droplets.
This field may be used to build a hierarchy of droplets in more complex excitations.

"""
mutable struct Droplet
    denergy::Real  # excitation energy
    first::Int  # site where droplet starts
    last::Int
    flip::Flip  # states of nodes flipped by droplet
    droplets::Union{NoDroplets,Vector{Droplet}}  # subdroplets on top of the current droplet; can be empty
end

Droplets = Union{NoDroplets,Vector{Droplet}}  # Can't be defined before Droplet struct

"""
$(TYPEDSIGNATURES)

This is a method used to calculate excitation information for the `NoDroplets` strategy in the context of a SpinGlassPEPS contractor.
The `NoDroplets` strategy represents a scenario in which no droplets are present in the system, and therefore, no excitation information is calculated.
    
## Arguments
- `method::NoDroplets`: An instance of the `NoDroplets` strategy.
- `ctr::MpsContractor{T}`: A SpinGlassPEPS contractor of type `T` representing the system.
- `best_idx::Int`: The index of the best state.
- `energies::Vector{<:Real}`: A vector of energies associated with different states.
- `states::Vector{Vector{Int}}`: A vector of states represented as arrays of integers.
- `droplets::Vector{Droplets}`: A vector of droplets in the system.
- `spins::Vector{Vector{Int}}`: A vector of spin configurations associated with states.

## Returns
- `NoDroplets()`: An instance of the `NoDroplets` strategy indicating that no excitation information is calculated in this scenario.
"""
(method::NoDroplets)(
    ::MpsContractor,
    ::Int,
    ::Vector{<:Real},
    ::Vector{Vector{Int}},
    ::Vector{Droplets},
    ::Vector{Vector{Int}},
) = NoDroplets()

# """
# $(TYPEDSIGNATURES)

# This method calculates excitation information for the `SingleLayerDroplets` strategy in the context of a SpinGlassPEPS contractor. 
# The `SingleLayerDroplets` strategy represents a scenario in which excitations are calculated for single-layer droplets.

# ## Arguments 
# - `method::SingleLayerDroplets`: An instance of the `SingleLayerDroplets` strategy.
# - `ctr::MpsContractor{T}`: A SpinGlassPEPS contractor of type `T` representing the system.
# - `best_idx::Int`: The index of the best state.
# - `energies::Vector{<:Real}`: A vector of energies associated with different states.
# - `states::Vector{Vector{Int}}`: A vector of states represented as arrays of integers.
# - `droplets::Vector{Droplets}`: A vector of droplets in the system.
# - `spins::Vector{Vector{Int}}`: A vector of spin configurations associated with states.

# ## Returns
# A new `Droplets` object representing the updated droplets based on the `SingleLayerDroplets` strategy
# """
function (method::SingleLayerDroplets)(
    ::MpsContractor,
    best_idx::Int,
    energies::Vector{<:Real},
    states::Vector{Vector{Int}},
    droplets::Vector{Droplets},
    spins::Vector{Vector{Int}},
)
    ndroplets = copy(droplets[best_idx])
    bstate = states[best_idx]
    benergy = energies[best_idx]
    bspin = spins[best_idx]

    for ind ∈ (1:best_idx-1..., best_idx+1:length(energies)...)
        flip_support = findall(bstate .!= states[ind])
        flip_state = states[ind][flip_support]
        flip_spinxor = bspin[flip_support] .⊻ spins[ind][flip_support]
        flip_statexor = bstate[flip_support] .⊻ states[ind][flip_support]
        flip = Flip(flip_support, flip_state, flip_spinxor, flip_statexor)
        denergy = energies[ind] - benergy
        droplet = Droplet(denergy, flip_support[1], length(bstate), flip, NoDroplets())
        if droplet.denergy <= method.max_energy &&
           hamming_distance(droplet.flip, method.mode) >= method.min_size
            ndroplets = my_push!(ndroplets, droplet, method)
        end
        for subdroplet ∈ droplets[ind]
            new_droplet = merge_droplets(method, droplet, subdroplet)
            if new_droplet.denergy <= method.max_energy &&
               hamming_distance(new_droplet.flip, method.mode) >= method.min_size
                ndroplets = my_push!(ndroplets, new_droplet, method)
            end
        end
    end
    if typeof(ndroplets) == NoDroplets
        return ndroplets
    else
        return filter_droplets(ndroplets, method)
    end
end

"""
$(TYPEDSIGNATURES)

Filter a vector of droplets based on specified criteria and strategy parameters.

## Arguments
- `all_droplets::Vector{Droplet}`: A vector of `Droplet` objects representing the droplets to be filtered.
- `method::SingleLayerDroplets`: An instance of the `SingleLayerDroplets` strategy used to determine filtering criteria.

## Returns
- `filtered_droplets::Vector{Droplet}`: A filtered vector of `Droplet` objects based on the specified criteria and strategy parameters.
"""
function filter_droplets(all_droplets::Vector{Droplet}, method::SingleLayerDroplets)
    sorted_droplets = sort(all_droplets, by = droplet -> (droplet.denergy))
    if method.metric == :hamming
        cutoff = method.min_size
    else #method.metric == :no_metric
        cutoff = -Inf
    end

    filtered_droplets = Droplet[]
    for droplet in sorted_droplets
        should_push = true
        for existing_drop in filtered_droplets
            if diversity_metric(existing_drop, droplet, method.metric, method.mode) < cutoff
                should_push = false
                break
            end
        end
        if should_push
            push!(filtered_droplets, droplet)
        end
    end
    filtered_droplets
end

"""
$(TYPEDSIGNATURES)

Push a 'Droplet' object into a vector of droplets ('Droplets') while considering the strategy parameters.

## Arguments
- `ndroplets::Droplets`: A vector of 'Droplet' objects to which the new 'Droplet' object will be added.
- `droplet::Droplet`: The 'Droplet' object to be added to the vector.
- `method`: The strategy parameter that determines whether or not the 'Droplet' object is added based on the defined criteria.

## Returns
- `ndroplets::Droplets`: The updated vector of 'Droplet' objects after the addition of the new 'Droplet' object.
"""
function my_push!(ndroplets::Droplets, droplet::Droplet, method)
    if typeof(ndroplets) == NoDroplets
        ndroplets = Droplet[]
    end
    push!(ndroplets, droplet)
    ndroplets
end

"""
$(TYPEDSIGNATURES)
Calculate the diversity metric between two 'Droplet' objects based on the specified metric.

## Arguments
- `drop1::Droplet`: The first 'Droplet' object for comparison.
- `drop2::Droplet`: The second 'Droplet' object for comparison.
- `metric::Symbol`: A symbol specifying the metric to be used for the diversity calculation. Currently, only the "hamming" metric is supported.

## Returns
- `d::Real`: The calculated diversity metric value between the two 'Droplet' objects.
"""
function diversity_metric(drop1::Droplet, drop2::Droplet, metric::Symbol, mode::Symbol)
    if metric == :hamming
        d = hamming_distance(drop1.flip, drop2.flip, mode)
    else
        d = Inf
    end
    d
end

"""
$(TYPEDSIGNATURES)

Calculate the Hamming distance for a 'Flip' object.
## Arguments
- `flip::Flip`: The 'Flip' object for which the Hamming distance will be calculated.
    
## Returns
- `d::Int`: The computed Hamming distance.
"""
hamming_distance(flip::Flip, s::Symbol) = hamming_distance(flip, Val(s))

hamming_distance(flip::Flip, ::Val{:Ising}) = sum(count_ones(st) for st ∈ flip.spinxor)

hamming_distance(flip::Flip, ::Val{:RMF}) = sum(count_ones(st) for st ∈ flip.statexor)

"""
$(TYPEDSIGNATURES)

Calculate the Hamming distance between two vectors of states.

## Arguments
- `state1::Vector{Int}`: The first vector.
- `state2::Vector{Int}`: The second vector.

## Returns
- `d::Int`: The computed Hamming distance.
"""
hamming_distance(state1, state2, s::Symbol) = hamming_distance(state1, state2, Val(s))

hamming_distance(state1, state2, ::Val{:Ising}) =
    sum(count_ones(st) for st ∈ state1 .⊻ state2)

# hamming_distance(state1, state2) = sum(state1 .!== state2)
hamming_distance(state1, state2, ::Val{:RMF}) = state1 == state2 ? 0 : 1

"""
$(TYPEDSIGNATURES)

Calculate the Hamming distance between two Flip objects representing states with support and flip information.

## Arguments
- `flip1::Flip`: The first Flip object, containing support, state, and spinxor information.
- `flip2::Flip`: The second Flip object, with support, state, and spinxor information.

## Returns
- `hd::Int`: The computed Hamming distance between the two Flip objects.
"""
hamming_distance(flip1::Flip, flip2::Flip, s::Symbol) =
    hamming_distance(flip1, flip2, Val(s))

function hamming_distance(flip1::Flip, flip2::Flip, ::Val{:Ising})
    n1, n2, hd = 1, 1, 0
    l1, l2 = length(flip1.support), length(flip2.support)
    while (n1 <= l1) && (n2 <= l2)
        if flip1.support[n1] == flip2.support[n2]
            if flip1.state[n1] != flip2.state[n2]
                hd += count_ones(flip1.spinxor[n1] ⊻ flip2.spinxor[n2])
            end
            n1 += 1
            n2 += 1
        elseif flip1.support[n1] < flip2.support[n2]
            hd += count_ones(flip1.spinxor[n1])
            n1 += 1
        else
            hd += count_ones(flip2.spinxor[n2])
            n2 += 1
        end
    end
    while n1 <= l1
        hd += count_ones(flip1.spinxor[n1])
        n1 += 1
    end
    while n2 <= l2
        hd += count_ones(flip2.spinxor[n2])
        n2 += 1
    end
    hd
end

function hamming_distance(flip1::Flip, flip2::Flip, ::Val{:RMF})
    n1, n2, hd = 1, 1, 0
    l1, l2 = length(flip1.support), length(flip2.support)
    while (n1 < l1) && (n2 < l2)
        if flip1.support[n1] == flip2.support[n2]
            if flip1.state[n1] != flip2.state[n2]
                hd += 1
            end
            n1 += 1
            n2 += 1
        elseif flip1.support[n1] < flip2.support[n2]
            n1 += 1
            hd += 1
        else
            n2 += 1
            hd += 1
        end
    end
    if n1 < l1
        hd += l1 - n1
    elseif n2 < l2
        hd += l2 - n2
    end
    hd
end

"""
$(TYPEDSIGNATURES)
Merge two Droplets according to the specified `SingleLayerDroplets` method.

## Arguments
- `method::SingleLayerDroplets`: The method used to determine whether and how to merge the droplets.
- `droplet::Droplet`: The main droplet to be merged.
- `subdroplet::Droplet`: The subdroplet to be merged with the main droplet.

## Returns
- `merged_droplet::Droplet`: The merged droplet created based on the merging method.
"""
function merge_droplets(method::SingleLayerDroplets, droplet::Droplet, subdroplet::Droplet)
    denergy = droplet.denergy + subdroplet.denergy
    first = min(droplet.first, subdroplet.first)
    last = max(droplet.last, subdroplet.last)

    flip = droplet.flip
    subflip = subdroplet.flip

    i1, i2, i3 = 1, 1, 1
    ln = length(union(flip.support, subflip.support))

    new_support = zeros(Int, ln)
    new_state = zeros(Int, ln)
    new_spinxor = zeros(Int, ln)
    new_statexor = zeros(Int, ln)

    while i1 <= length(flip.support) && i2 <= length(subflip.support)
        if flip.support[i1] == subflip.support[i2]
            new_support[i3] = flip.support[i1]
            new_state[i3] = subflip.state[i2]
            new_spinxor[i3] = flip.spinxor[i1] ⊻ subflip.spinxor[i2]
            new_statexor[i3] = flip.statexor[i1] ⊻ subflip.statexor[i2]
            i1 += 1
            i2 += 1
            i3 += 1
        elseif flip.support[i1] < subflip.support[i2]
            new_support[i3] = flip.support[i1]
            new_state[i3] = flip.state[i1]
            new_spinxor[i3] = flip.spinxor[i1]
            new_statexor[i3] = flip.statexor[i1]
            i1 += 1
            i3 += 1
        else # flip.support[i1] > subflip.support[i2]
            new_support[i3] = subflip.support[i2]
            new_state[i3] = subflip.state[i2]
            new_spinxor[i3] = subflip.spinxor[i2]
            new_statexor[i3] = subflip.statexor[i2]
            i2 += 1
            i3 += 1
        end
    end
    while i1 <= length(flip.support)
        new_support[i3] = flip.support[i1]
        new_state[i3] = flip.state[i1]
        new_spinxor[i3] = flip.spinxor[i1]
        new_statexor[i3] = flip.statexor[i1]
        i1 += 1
        i3 += 1
    end
    while i2 <= length(subflip.support)
        new_support[i3] = subflip.support[i2]
        new_state[i3] = subflip.state[i2]
        new_spinxor[i3] = subflip.spinxor[i2]
        new_statexor[i3] = subflip.statexor[i2]
        i2 += 1
        i3 += 1
    end
    flip = Flip(new_support, new_state, new_spinxor, new_statexor)
    Droplet(denergy, first, last, flip, NoDroplets())
end

"""
$(TYPEDSIGNATURES)

Apply a flip operation to a state.

## Arguments
- `state::Vector{Int}`: The original state vector.
- `flip::Flip`: The flip operation to be applied to the state.

## Returns
- `new_state::Vector{Int}`: The modified state after applying the flip operation.
"""
function flip_state(state::Vector{Int}, flip::Flip)
    new_state = copy(state)
    new_state[flip.support] .= flip.state
    new_state
end

"""
$(TYPEDSIGNATURES)

Unpack droplets in a solution structure to create a new solution with individual excitations.

## Arguments
- `sol`: The input solution containing droplets to be unpacked.
- `β::Real`: The inverse temperature parameter used for probability adjustments.

## Returns
- `new_sol`: A new solution where droplets are unpacked into individual excitations.
"""
function unpack_droplets(sol, β)  # have β in sol ?
    energies = typeof(sol.energies[begin])[]
    states = typeof(sol.states[begin])[]
    probs = typeof(sol.probabilities[begin])[]
    degeneracy = typeof(sol.degeneracy[begin])[]
    droplets = Droplets[]
    spins = typeof(sol.spins[begin])[]

    for i = 1:length(sol.energies)
        push!(energies, sol.energies[i])
        push!(states, sol.states[i])
        push!(probs, sol.probabilities[i])
        push!(degeneracy, 1)
        push!(droplets, NoDroplets())
        push!(spins, sol.spins[i])

        for droplet in sol.droplets[i]
            push!(energies, sol.energies[i] + droplet.denergy)
            push!(states, flip_state(sol.states[i], droplet.flip))
            push!(probs, sol.probabilities[i] - β * droplet.denergy)
            push!(degeneracy, 1)
            push!(droplets, NoDroplets())
            push!(spins, sol.spins[i])
        end
    end

    inds = sortperm(energies)
    Solution(
        energies[inds],
        states[inds],
        probs[inds],
        degeneracy[inds],
        sol.largest_discarded_probability,
        droplets[inds],
        spins[inds],
    )
end

"""
$(TYPEDSIGNATURES)
Apply a permutation to a 'NoDroplets' object, resulting in an unchanged 'NoDroplets'.

## Arguments
- `drop::NoDroplets`: The 'NoDroplets' object that remains unchanged.
- `perm::Vector{Int}`: A permutation vector that is applied to indices.

## Returns
- `result::NoDroplets`: The 'NoDroplets' object, which remains the same.
"""
perm_droplet(drop::NoDroplets, perm::Vector{Int}) = drop

"""
$(TYPEDSIGNATURES)
Apply a permutation to a collection of 'Droplet' objects.

## Arguments
- `drops::Vector{Droplet}`: A vector of 'Droplet' objects to which the permutation is applied.
- `perm::Vector{Int}`: A permutation vector that is applied to indices.

## Returns
- `result::Vector{Droplet}`: A vector of 'Droplet' objects after applying the permutation.

"""
function perm_droplet(drops::Vector{Droplet}, perm::Vector{Int})
    [perm_droplet(drop, perm) for drop in drops]
end

"""
$(TYPEDSIGNATURES)
Apply a permutation to a 'Droplet' object.

## Arguments
- `drop::Droplet`: A 'Droplet' object to which the permutation is applied.
- `perm::Vector{Int}`: A permutation vector that is applied to indices.

## Returns
- `result::Droplet`: A 'Droplet' object after applying the permutation.
"""
function perm_droplet(drop::Droplet, perm::Vector{Int})
    flip = drop.flip
    support = perm[flip.support]
    ind = sortperm(support)
    flip = Flip(support[ind], flip.state[ind], flip.spinxor[ind], flip.statexor[ind])
    Droplet(drop.denergy, drop.first, drop.last, flip, perm_droplet(drop.droplets, perm))
end
