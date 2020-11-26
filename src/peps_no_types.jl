using TensorOperations
using LinearAlgebra
using GenericLinearAlgebra
using Distributed


# TODO β and interactions should be as Float64, if typechange, make it inside a solver

function get_parameters_for_T(g::MetaGraph, i::Int)
    no_spins = length(props(g, i)[:spins])
    tensor_size = [1,1,1,1, 2^no_spins]
    right = Int[]
    down = Int[]
    M_left = zeros(1, 2^no_spins)
    M_up = zeros(1, 2^no_spins)
    for n in all_neighbors(g, i)
        if props(g, i)[:column] -1 == props(g, n)[:column]
            M_left = props(g, i, n)[:M]
            tensor_size[1] = size(M_left, 1)
        elseif props(g, i)[:row] -1 == props(g, n)[:row]
            M_up = props(g, i, n)[:M]
            tensor_size[2] = size(M_up, 1)

        elseif props(g, i)[:column] +1 == props(g, n)[:column]
            right = props(g, i, n)[:inds]
            tensor_size[3] = 2^length(right)
        elseif props(g, i)[:row] +1 == props(g, n)[:row]
            down = props(g, i, n)[:inds]
            tensor_size[4] = 2^length(down)
        end
    end
    no_spins, tensor_size, right, down, M_left, M_up
end

"""
compute_single_tensor(g::MetaGraph, i::Int, β::T; sum_over_last::Bool = false) where T <: AbstractFloat

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


function compute_single_tensor(g::MetaGraph, i::Int, β::T; sum_over_last::Bool = false) where T <: AbstractFloat
    n = 0
    no_spins, tensor_size, right, down, M_left, M_up = get_parameters_for_T(g, i)

    tensor = zeros(T, (tensor_size[1:4]...))
    if !sum_over_last
        tensor = zeros(T, (tensor_size...))
    end

    column = props(g, i)[:column]
    row = props(g, i)[:row]
    log_energy = props(g, i)[:energy]

    k1 = [reindex(i, no_spins, right) for i in 1:tensor_size[5]]
    k2 = [reindex(i, no_spins, down) for i in 1:tensor_size[5]]

    for k in CartesianIndices(tuple(tensor_size[1], tensor_size[2]))
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
        for i in 1:tensor_size[5]


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
    function compute_scalar_prod(mps_down::MPS{T}, mps_up::MPS{T}) where T <: AbstractFloat

Returns matrix, the scalar product of two mpses, with open two legs of first elements.
"""
function compute_scalar_prod(mps_down::MPS{T}, mps_up::MPS{T}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    env
end

"""
    function scalar_prod_step(mps_down::Array{T, 3}, mps_up::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat

Returns matrix
"""

function scalar_prod_step(mps_down::Array{T, 3}, mps_up::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1))

    @tensor begin
        C[a,b] = mps_up[a,z,x]*mps_down[b,z,y]*env[x,y]
    end
    C
end


"""
    function set_spin_from_letf(mpo::Vector{Array{T,4}}, new_s::Int) where T <: AbstractFloat

Given mpo, returns a vector of 3-mode arrays

First is the l th element of mpo where
first mode index is set to new_s (this is the configuration of l-1 th element).

Further are traced over the physical (last) dimension.
"""
function set_spin_from_letf(mpo::Vector{Array{T,4}}, new_s::Int) where T <: AbstractFloat
    B = mpo[1][new_s,:,:,:]
    B = permutedims(B, (3,1,2))
    mps = vcat([B], [sum_over_last(el) for el in mpo[2:end]])
    MPS(mps)
end


function Mprod(Ms::Vector{Array{T, 2}}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for M in Ms
        env = env*M
    end
    env
end


function make_lower_mps(g::MetaGraph, k::Int, β::T, χ::Int, threshold::Float64) where T <: AbstractFloat
    grid = props(g)[:grid]
    s = size(grid,1)
    mps = MPS([ones(T, (1,1,1)) for _ in 1:size(grid,2)])

    for i in s:-1:k
        mpo = [compute_single_tensor(g, j, β; sum_over_last = true) for j in grid[i,:]]
        mps = MPO(mpo)*mps
        if threshold > 0.
            mps = compress(mps, χ, threshold)
        end
    end
    return mps
end

"""
    mutable struct Partial_sol{T <: AbstractFloat}

structure of the partial solution
"""
mutable struct Partial_sol{T <: AbstractFloat}
    spins::Vector{Int}
    objective::T
    function(::Type{Partial_sol{T}})(spins::Vector{Int}, objective::T) where T <:AbstractFloat
        new{T}(spins, objective)
    end
    function(::Type{Partial_sol{T}})() where T <:AbstractFloat
        new{T}(Int[], 1.)
    end
end


"""
    update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat

Add a spin and replace an objective function to Partial_sol{T} type
"""
# TODO move particular type to solver
function update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat
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
        for i in column:s[2]
            k = grid[row-1,i]
            k1 = grid[row,i]
            all = props(gg, k)[:spins]

            index = ps.spins[k]
            ind = props(gg, k, k1)[:inds]
            upper_right[i-column+1] = reindex(index, length(all), ind)
        end
    end
    if row < s[1]
        for i in 1:column-1
            k = grid[row,i]
            k1 = grid[row+1,i]
            all = props(gg, k)[:spins]
            ind = props(gg, k, k1)[:inds]
            index = ps.spins[k]

            upper_left[i] = reindex(index, length(all), ind)
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
        ind = props(gg, j, jp)[:inds]
        return reindex(ps.spins[end], length(all), ind)
    end
    1
end

function conditional_probabs(gg::MetaGraph, ps::Partial_sol{T}, j::Int, lower_mps::MPS{T},
                                            vec_of_T::Vector{Array{T,5}}) where T <: AbstractFloat

    upper_left, upper_right = spin_indices_from_above(gg, ps, j)
    left_s = spin_index_from_left(gg, ps, j)
    l = props(gg, j)[:column]
    grid = props(gg)[:grid]

    M = [vec_of_T[k][:,upper_right[k-l+1],:,:,:] for k in l:size(grid,2)]
    # move to mps notation
    M = [permutedims(e, (1,3,2,4)) for e in M]
    upper_mps = set_spin_from_letf(M, left_s)

    partial_scalar_prod = compute_scalar_prod(MPS(lower_mps[l:end]), upper_mps)
    
    lower_mps_left = [lower_mps[i][:,upper_left[i],:] for i in 1:l-1]
    weight = Mprod(lower_mps_left)
    probs_unnormed = partial_scalar_prod*transpose(weight)

    probs_unnormed./sum(probs_unnormed)
end


function solve(g::MetaGraph, no_sols::Int = 2; β::T, χ::Int = 0,
                threshold::Float64 = 1e-14, node_size::Tuple{Int, Int} = (1,1)) where T <: AbstractFloat

    gg = graph4peps(g, node_size)

    grid = props(gg)[:grid]

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:size(grid,1)
        @info "row of peps = " row
        #this may be cashed
        lower_mps = make_lower_mps(gg, row + 1, β, χ, threshold)

        vec_of_T = [compute_single_tensor(gg, j, β) for j in grid[row,:]]

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[]
            for ps in partial_s

                objectives = conditional_probabs(gg, ps, j, lower_mps, vec_of_T)

                for l in 1:length(objectives)
                    push!(partial_s_temp, update_partial_solution(ps, l, ps.objective*objectives[l]))
                end

            end
            partial_s = select_best_solutions(partial_s_temp, no_sols)

            if j == maximum(grid)
                return return_solutions(partial_s, gg)
            end
        end
    end
end

"""
    return_solutions(partial_s::Vector{Partial_sol{T}})

return final solutions sorted backwards in form Vector{Partial_sol{T}}
spins are given in -1,1
"""
function return_solutions(partial_s::Vector{Partial_sol{T}}, ns:: MetaGraph)  where T <: AbstractFloat

    l = length(partial_s)
    objective = zeros(T, l)
    spins = [Int[] for _ in 1:l]
    size = get_system_size(ns)
    # order is reversed, to correspond with sort
    for i in 1:l
        one_solution = zeros(Int, size)
        objective[l-i+1] = partial_s[i].objective

        ses = partial_s[i].spins

        for k in vertices(ns)
            spins_inds = props(ns, k)[:spins]
            ii = ind2spin(ses[k], length(spins_inds))
            for j in 1:length(ii)
                one_solution[spins_inds[j]] = ii[j]
            end
        end
        spins[l-i+1] = one_solution
    end

    return spins, objective
end

"""
    select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:AbstractFloat

returns Vector{Partial_sol{T}}, a vector of no_sols best solutions
"""
function select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:AbstractFloat
    obj = [ps.objective for ps in partial_s_temp]
    perm = sortperm(obj)
    p = last_m_els(perm, no_sols)
    return partial_s_temp[p]
end
