
# TODO β and interactions should be as Float64, if typechange, make it inside a solver

function get_parameters_for_T(g::MetaGraph, i::Int)
    no_spins = length(props(g, i)[:spins])
    spectrum = props(g, i)[:spectrum]

    right = Int[]
    down = Int[]
    #M_left = zeros(1, 2^no_spins)
    #M_up = zeros(1, 2^no_spins)
    M_left = zeros(1, length(spectrum))
    M_up = zeros(1, length(spectrum))

    for n ∈ all_neighbors(g, i)

        if props(g, i)[:column] -1 == props(g, n)[:column]

            spectrum = props(g, n)[:spectrum]
            ind = props(g, i, n)[:inds]
            rrr = [s[ind] for s ∈ spectrum]
            k1 = unique([spins2ind(e) for e ∈ rrr])
            p = invperm(sortperm(k1))
            M_left = props(g, i, n)[:M][p,:]

        elseif props(g, i)[:row] -1 == props(g, n)[:row]
            spectrum = props(g, n)[:spectrum]
            ind = props(g, i, n)[:inds]
            rrr = [s[ind] for s ∈ spectrum]
            k1 = unique([spins2ind(e) for e ∈ rrr])
            p = invperm(sortperm(k1))
            M_up = props(g, i, n)[:M][p,:]


        elseif props(g, i)[:column] +1 == props(g, n)[:column]
            right = props(g, i, n)[:inds]

        elseif props(g, i)[:row] +1 == props(g, n)[:row]
            down = props(g, i, n)[:inds]

        end
    end
    right, down, M_left, M_up
end

"""
compute_single_tensor(g::MetaGraph, i::Int, β::T; sum_over_last::Bool = false) where T <: Real

Returns tensors, building blocks for a peps initialy tensor is 5 mode:

            5 .    2
                .  |
                  .|
            1 ---  T ---- 3
                   |
                   |
                   4
mode 5 is physical.


If sum_over_last -- summed over mode 5
"""


function compute_single_tensor(g::MetaGraph, i::Int, β::T; sum_over_last::Bool = false) where T <: Real
    n = 0
    right, down, M_left, M_up = get_parameters_for_T(g, i)

    column = props(g, i)[:column]
    row = props(g, i)[:row]
    log_energy = props(g, i)[:energy]
    spectrum = props(g, i)[:spectrum]

    tensor_size = [size(M_left, 1), size(M_up, 1) ,1,1, length(spectrum)]


    r = [s[right] for s ∈ spectrum]
    k1 = s2i(r)
    tensor_size[3] = maximum(k1)

    d = [s[down] for s ∈ spectrum]
    k2 = s2i(d)
    tensor_size[4] = maximum(k2)

    tensor = zeros(T, (tensor_size[1:4]...))
    if !sum_over_last
        tensor = zeros(T, (tensor_size...))
    end

    for k ∈ CartesianIndices(tuple(tensor_size[1], tensor_size[2]))
        energy = log_energy

        # conctraction with Ms
        if column > 1
            @inbounds energy = energy + M_left[k[1], :]
        end

        if row > 1
            @inbounds energy = energy + M_up[k[2], :]
        end
        energy = exp.(-β.*(energy))


        # itteration over physical index


        for i ∈ 1:tensor_size[5]

            if !sum_over_last
                @inbounds tensor[k[1], k[2], k1[i], k2[i], i] = energy[i]
            else
                @inbounds tensor[k[1], k[2], k1[i], k2[i]] = tensor[k[1], k[2], k1[i], k2[i]] + energy[i]
            end
        end
    end
    return tensor
end


"""
    function set_spin_from_letf(mpo::AbstractMPO{T}, new_s::Int) where T <: Real

Given mpo, returns a vector of 3-mode arrays

First is the l th element of mpo where
first mode index is set to new_s (this is the configuration of l-1 th element).

Further are traced over the physical (last) dimension.
"""
function set_spin_from_letf(mpo::AbstractMPO{T}, new_s::Int) where T <: Real
    B = mpo[1][new_s,:,:,:]
    B = permutedims(B, (3,1,2))
    mps = vcat([B], [sum_over_last(el) for el ∈ mpo[2:end]])
    MPS(mps)
end

"""
    set_spins_from_above(vec_of_T::Vector{Array{T,5}}, upper_right::Vector{Int})
"""
#      upper_right
# α.     |                    |
#    .   |                    |
#    --- M ----   =>     ----- M ------
#        |                    |
#        |                    |α

function set_spins_from_above(vec_of_T::Vector{Array{T,5}}, upper_right::Vector{Int}) where T <: Real
    l = length(vec_of_T) - length(upper_right)+1
    M = [vec_of_T[k][:,upper_right[k-l+1],:,:,:] for k ∈ l:length(vec_of_T)]
    MPO([permutedims(e, (1,3,2,4)) for e ∈ M])
end

function make_lower_mps(g::MetaGraph, k::Int, β::T, χ::Int, threshold::Float64) where T <: Real
    grid = props(g)[:grid]
    s = size(grid,1)
    mps = MPS([ones(T, (1,1,1)) for _ ∈ 1:size(grid,2)])

    for i ∈ s:-1:k
        mpo = [compute_single_tensor(g, j, β; sum_over_last = true) for j ∈ grid[i,:]]
        mps = MPO(mpo)*mps
        if χ < size(mps[1], 3)
            mps = compress(mps, χ, threshold)
        end
    end
    return mps
end

"""
    mutable struct Partial_sol{T <: Real}

structure of the partial solution
"""
mutable struct Partial_sol{T <: Real}
    spins::Vector{Int}
    objective::T

    function(::Type{Partial_sol{T}})(spins::Vector{Int}, objective::T) where T <:Real
        new{T}(spins, objective)
    end
    function(::Type{Partial_sol{T}})() where T <:Real
        new{T}(Int[], 1.)
    end
end


"""
    update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: Real

Add a spin and replace an objective function to Partial_sol{T} type
"""
# TODO move particular type to solver
function update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: Real
    Partial_sol{T}(vcat(ps.spins, [s]), objective)
end

"""
    spin_indices_from_above(gg::MetaGraph, ps::Partial_sol, j::Int)

returns two vectors of incdices from above to the cutoff.

              physical .
                         .   upper_right
                           .  |      |
    upper_left   from_left-- A3 --  A4 -- 1
      |    |                 |      |
1 -- B1 -- B2       --       B3  -- B4 -- 1
"""
function spin_indices_from_above(gg::MetaGraph, ps::Partial_sol, j::Int)
    grid = props(gg)[:grid]
    s = size(grid)
    row = props(gg, j)[:row]
    column = props(gg, j)[:column]


    upper_right = ones(Int, s[2]-column+1)
    upper_left = ones(Int, column-1)

    if row > 1
        for i ∈ column:s[2]
            k = grid[row-1,i]
            k1 = grid[row,i]
            all = props(gg, k)[:spins]
            spectrum = props(gg, k)[:spectrum]

            index = ps.spins[k]
            ind = props(gg, k, k1)[:inds]

            d = [s[ind] for s ∈ spectrum]
            k2 = s2i(d)

            upper_right[i-column+1] = k2[index]
        end
    end
    if row < s[1]
        for i ∈ 1:column-1
            k = grid[row,i]
            k1 = grid[row+1,i]

            all = props(gg, k)[:spins]
            ind = props(gg, k, k1)[:inds]
            spectrum = props(gg, k)[:spectrum]

            index = ps.spins[k]

            d = [s[ind] for s ∈ spectrum]
            k2 = s2i(d)

            upper_left[i] = k2[index]
        end
    end
    upper_left, upper_right
end


function spin_index_from_left(gg::MetaGraph, ps::Partial_sol, j::Int)
    grid = props(gg)[:grid]
    column = props(gg, j)[:column]
    row = props(gg, j)[:row]

    if  column > 1
        jp = grid[row, column-1]
        all = props(gg, jp)[:spins]
        spectrum = props(gg, jp)[:spectrum]
        ind = props(gg, j, jp)[:inds]

        d = [s[ind] for s ∈ spectrum]
        k2 = s2i(d)
        return k2[ps.spins[end]]
    end
    1
end

function conditional_probabs(gg::MetaGraph, ps::Partial_sol{T}, j::Int, lower_mps::AbstractMPS{T},
                                            vec_of_T::Vector{Array{T,5}}) where T <: Real

    upper_left, upper_right = spin_indices_from_above(gg, ps, j)
    left_s = spin_index_from_left(gg, ps, j)
    l = props(gg, j)[:column]
    grid = props(gg)[:grid]

    upper_mpo = set_spins_from_above(vec_of_T, upper_right)
    upper_mps = set_spin_from_letf(upper_mpo, left_s)
    re = right_env(MPS(lower_mps[l:end]), upper_mps)[1]

    weight = ones(T, 1,1)
    if l > 1
        Mat = [lower_mps[i][:,upper_left[i],:] for i ∈ 1:l-1]
        weight = prod(Mat)
    end
    probs_unnormed = re*transpose(weight)

    probs_unnormed./sum(probs_unnormed)
end

"""
    function dX_inds(grid::Matrix{Int}, j::Int; has_diagonals::Bool = false)

Returns vector{Int} indexing of the boundary region (dX) given a grid.
id has diagonals, diagonal bounds on the grid are taken into account
"""

function dX_inds(s::Int, j::Int; has_diagonals::Bool = false)
    last = j-1
    first = maximum([1, j - s])
    if (has_diagonals & (j%s != 1))
        first = maximum([1, j - s - 1])
    end
    return collect(first: last)
end

"""
    function merge_dX(partial_s::Vector{Partial_sol{T}}, dX_inds::Vector{Int}, δH::Float64) where T <:Real

Return a vector of Partial_sol{T}, with merged boundaries.

Merging rule is such that the retion of the objective function of the merged item
to the maximal is lower than δH
"""

function merge_dX(partial_s::Vector{Partial_sol{T}}, dX_inds::Vector{Int}, δH::Float64) where T <:Real
    if (length(partial_s) > 1) & (δH != .0)
        leave = [true for _ ∈ partial_s]

        dXes = [ps.spins[dX_inds] for ps ∈ partial_s]

        unique_dXes = unique(dXes)
        if dXes != unique_dXes
            dXcount = countmap(dXes)
            for dX ∈ unique_dXes
                if dXcount[dX] > 1
                    i = findall(k -> k == dX, dXes)
                    objectives = [partial_s[j].objective for j ∈ i]

                    objectives = objectives./maximum(objectives)
                    for ind ∈ i[objectives .< δH]
                        leave[ind] = false
                    end
                end
            end
            no_reduced = count(.!(leave))
            # this is just for testing
            if no_reduced > 0
                j = length(partial_s[1].spins)
                k = length(partial_s)
                println(no_reduced, " out of $k partial solutions deleted at j = $j")
            end
            return partial_s[leave]
        end
    end
    partial_s
end


function solve(g::MetaGraph, peps::PepsNetwork, no_sols::Int = 2; node_size::Tuple{Int, Int} = (1,1),
                                               β::T, χ::Int = 2^prod(node_size),
                                               threshold::Float64 = 0.,
                                               spectrum_cutoff::Int = 1000,
                                               δH::Float64 = 0.) where T <: Real


    gg = graph4peps(g, node_size, spectrum_cutoff = spectrum_cutoff)
    max_sweeps=4

    boundary_mps = boundaryMPS(peps, χ, threshold, max_sweeps)

    grid = props(gg)[:grid]

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row ∈ 1:peps.i_max
        @info "row of peps = " row

        lower_mps = make_lower_mps(gg, row + 1, β, χ, threshold)

        vec_of_T = [compute_single_tensor(gg, j, β) for j ∈ grid[row,:]]

        peps_row = PEPSRow(peps, row)

        a = (row-1)*peps.j_max+1
        b = row*peps.j_max
        #println([x for x in a:1:b] - grid[row,:])
        for j ∈ a:1:b

            dX = dX_inds(peps.j_max, j)

            partial_s_temp = Partial_sol{T}[]
            # TODO better compare energies, think it over
            partial_s = merge_dX(partial_s, dX, δH)
            for ps ∈ partial_s

                objectives = conditional_probabs(gg, ps, j, lower_mps, vec_of_T)

                for l ∈ eachindex(objectives)
                    new_objectives = ps.objective*objectives[l]
                    # TODO use log of probabilities
                    ps1 = update_partial_solution(ps, l, new_objectives)
                    push!(partial_s_temp, ps1)
                end

            end
            partial_s = select_best_solutions(partial_s_temp, no_sols)

            if j == peps.i_max*peps.j_max
                return return_solutions(partial_s, gg)
            end
        end
    end
end

"""
    return_solutions(partial_s::Vector{Partial_sol{T}})

return final solutions sorted backwards ∈ form Vector{Partial_sol{T}}
spins are given ∈ -1,1
"""
function return_solutions(partial_s::Vector{Partial_sol{T}}, ns:: MetaGraph)  where T <: Real

    l = length(partial_s)
    objective = zeros(T, l)
    spins = [Int[] for _ ∈ 1:l]
    size = get_system_size(ns)
    # order is reversed, to correspond with sort
    for i ∈ 1:l
        one_solution = zeros(Int, size)
        objective[l-i+1] = partial_s[i].objective

        ses = partial_s[i].spins

        for k ∈ vertices(ns)
            spins_inds = props(ns, k)[:spins]
            spectrum = props(ns, k)[:spectrum]
            iii = spectrum[ses[k]]

            for j ∈ eachindex(iii)
                one_solution[spins_inds[j]] = iii[j]
            end
        end
        spins[l-i+1] = one_solution
    end

    return spins, objective
end

"""
    select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:Real

returns Vector{Partial_sol{T}}, a vector of no_sols best solutions
"""
function select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:Real
    obj = [ps.objective for ps ∈ partial_s_temp]
    # TODO change sortperm to partial sort
    perm = sortperm(obj)
    p = last_m_els(perm, no_sols)

    return partial_s_temp[p]
end
