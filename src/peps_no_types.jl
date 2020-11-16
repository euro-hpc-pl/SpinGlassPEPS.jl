using TensorOperations
using LinearAlgebra
using GenericLinearAlgebra
using SharedArrays
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
        elseif props(g, i)[:column] +1 == props(g, n)[:column]
            right = props(g, i, n)[:inds]
            tensor_size[2] = 2^length(right)
        elseif props(g, i)[:row] -1 == props(g, n)[:row]
            M_up = props(g, i, n)[:M]
            tensor_size[3] = size(M_up, 1)
        elseif props(g, i)[:row] +1 == props(g, n)[:row]
            down = props(g, i, n)[:inds]
            tensor_size[4] = 2^length(down)
        end
    end
    no_spins, tensor_size, right, down, M_left, M_up
end

"""
compute_single_tensor(ns::Node_of_grid, β::T)

Returns an tensor form which mpses and mpos are build, initialy tensor is 5 mode:

            5 .    3
                .  |
                  .|
            1 ---  T ---- 2
                   |
                   |
                   4
and mode 5 is physical.

If tensor is expected to be on the top of the peps mode 3 is trivial and is removed
If tensor is expected to be on the bottom of the peps mode 4 is trivial and is removed
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
    log_energy = props(g, i)[:log_energy]
    for k in CartesianIndices(tuple(tensor_size[1], tensor_size[3]))
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

            # this is for δs
            k1 = reindex(i, no_spins, right)
            k2 = reindex(i, no_spins, down)

            if !sum_over_last
                @inbounds tensor[k[1], k1, k[2], k2, i] = energy[i]
            else
                @inbounds tensor[k[1], k1, k[2], k2] = tensor[k[1], k1, k[2], k2] + energy[i]
            end
        end
    end

    if length(down) == 0
        return dropdims(tensor, dims = 4)
    elseif row == 1
        return dropdims(tensor, dims = 3)
    end
    return tensor
end

"""
    MPSxMPO(mps_down::Vector{Array{T, 3}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat

returns an mps, the product of the mpo and mps
"""
# TODO Take from LP+BG code, or at lest simplify, remove reshape
function MPSxMPO(mps_down::Vector{Array{T, 3}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
        mps_res = Array{Union{Nothing, Array{T}}}(nothing, length(mps_down))
        for i in 1:length(mps_down)
        A = mps_down[i]
        B = mps_up[i]
        sa = size(A)
        sb = size(B)

        C = zeros(T, sa[1] , sb[1], sa[2], sb[2], sb[3])
        @tensor begin
            C[a,d,b,e,f] = A[a,b,x]*B[d,e,f,x]
        end
        mps_res[i] = reshape(C, (sa[1]*sb[1], sa[2]*sb[2], sb[3]))
    end
    Array{Array{T, 3}}(mps_res)
end

"""
    function compute_scalar_prod(mps_down::Vector{Array{T, N} where N}, mps_up::Vector{Array{T, 3}}) where T <: AbstractFloat

Returns vector, the scalar product of two mpses. The first elemnt of mps_down
is supposed to be a matrix, as its first mode is set due to the spin to the left.
The first elemnt of mps_up has two virtual modes and one physical. Ather elements have
all virtual modes.
"""
# this is Type dependent, leave it as it is
function compute_scalar_prod(mps_down::Vector{Array{T, N} where N}, mps_up::Vector{Array{T, 3}}) where T <: AbstractFloat
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
        C[a,b] = mps_up[a,x,z]*mps_down[b,y,z]*env[x,y]
    end
    C
end

"""
    function scalar_prod_step(mps_down::Array{T, 2}, mps_up::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat

Returns a vector, proportional to the vector of probabilities
"""
function scalar_prod_step(mps_down::Array{T, 2}, mps_up::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 3))

    @tensor begin
        C[a] = mps_up[x,z,a]*mps_down[y,z]*env[x,y]
    end
    C
end

"""
    function set_spin_from_letf(mps::Vector{Array{T,3}}, sol::Vector{Int}, s::Int) where T <: AbstractFloat

Returns a vector of matrices. First matrix is the corresponding mps with first
mode set to the value correspodning to the spin of the element to the left.
Following matrices are computed from mps by tracing out physical mode

"""
function set_spin_from_letf(mps::Vector{Array{T,3}}, l::Int, new_s::Int, max_l::Int) where T <: AbstractFloat

    Ms = []
    # no element to the left
    if l == 0
        # transposition is to make a simple matrix multiplication for computing a probability
        Ms = [Array(transpose(mps[1][1,:,:]))]
    else
        Ms = [Array(transpose(mps[1][new_s,:,:]))]
    end
    # otherwise no elements to the right
    if l < max_l
        Ms = vcat(Ms, [sum_over_last(el) for el in mps[2:end]])
    end
    return Ms
end

"""
    function set_spin_from_letf(mpo::Vector{Array{T,4}}, sol::Vector{Int}, s::Int) where T <: AbstractFloat

Returns a vector of 3 mode arrays computed from the mpo. For the first one the
first dim is set according to the spoin to the left. For next, the physical dimesion
is traced.
"""
function set_spin_from_letf(mpo::Vector{Array{T,4}}, l::Int, new_s::Int, max_l::Int) where T <: AbstractFloat
    #l = length(sol)
    B = []
    if l == 0
        B = [mpo[1][1,:,:,:]]
    else
        B = [mpo[1][new_s,:,:,:]]
    end
    if l < max_l
        B = vcat(B, [sum_over_last(el) for el in mpo[2:end]])
    end
    return B
end

# TODO, rename and explain s, s_new it was done temporarly
# explain tensors B,C,D
function conditional_probabs(M::Vector{Array{T,4}}, lower_mps::Vector{Array{T,3}}, new_s::Int, s::Vector{Int} = Int[]) where T <: AbstractFloat

    l1 = length(lower_mps)
    l = length(s)

    A = set_spin_from_letf(M[l+1:end], l, new_s, l1-1)

    D = 0
    if l > 0
        B = ones(T, 1)
        for i in 1:l

            B = transpose(B)*lower_mps[i][:,:,s[i]]
            B = B[1,:]
        end
        E = lower_mps[l+1]

        @tensor begin
            C[b,c] := E[a,b,c]*B[a]
        end
        D = vcat([C], [lower_mps[i] for i in l+2:l1])
    else
        D = vcat([lower_mps[1][1,:,:]], [lower_mps[i] for i in l+2:l1])
    end

    unnorm_prob = compute_scalar_prod(D, A)
    unnorm_prob./sum(unnorm_prob)
end

"""
    function conditional_probabs(mps::Vector{Array{T, 2}}) where T <: AbstractFloat

Copmutes a conditional probability of the single mps (a last step)
"""
function conditional_probabs(mps::Vector{Array{T, 2}}) where T <: AbstractFloat
    env = ones(T, 1)
    for i in length(mps):-1:1
        env = mps[i]*env
    end
    env./sum(env)
end


function make_lower_mps(g::MetaGraph, k::Int, β::T, χ::Int, threshold::Float64) where T <: AbstractFloat
    grid = props(g)[:grid]
    s = size(grid,1)
    if k <= s
        mps = [compute_single_tensor(g, j, β; sum_over_last = true) for j in grid[s,:]]
        if threshold > 0.
            mps = compress_iter(mps, χ, threshold)
        end
        for i in s-1:-1:k

            #mpo = [sum_over_last(compute_single_tensor(ns[j], β)) for j in grid[i,:]]
            mpo = [compute_single_tensor(g, j, β; sum_over_last = true) for j in grid[i,:]]
            if threshold > 0.
                mpo = compress_iter(mpo, χ, threshold)
            end
            mps = MPSxMPO(mps, mpo)
            if threshold > 0.
                mps = compress_iter(mps, χ, threshold)
            end
        end
        return mps
        end
    [zeros(T,1)]
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

# TODO this will be the exported function

function solve(g::MetaGraph, no_sols::Int = 2; β::T, χ::Int = 0,
                threshold::Float64 = 1e-14, node_size::Tuple{Int, Int} = (1,1)) where T <: AbstractFloat

    gg = interactions2grid_graph(g, node_size)
    # grid follows the iiteration
    grid = props(gg)[:grid]
    #ns = 0.
    s = size(grid)
    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:s[1]
        println("row of peps = ", row)
        #this may need to ge cashed
        lower_mps = make_lower_mps(gg, row + 1, β, χ, threshold)

        upper_mpo = [compute_single_tensor(gg, j, β) for j in grid[row,:]]

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[]
            for ps in partial_s
                sol = ps.spins[1+(row-1)*s[2]:end]

                objectives = [T(0.)]
                # left cutoff
                new_s = 0
                # TODO it will be done on the grid coordinates of the node
                if props(gg, j)[:column] > 1
                    all = props(gg, j-1)[:spins]
                    ind = props(gg, j, j-1)[:inds]
                    new_s = reindex(sol[end], length(all), ind)

                end

                # reindex to contrtact with belowe
                if row < s[1]
                    for i in 1:length(sol)
                        k = grid[row,i]
                        k1 = grid[row+1,i]
                        #all = ns[k].spins_inds
                        all = props(gg, k)[:spins]

                        #ind = ns[k].down
                        ind = props(gg, k, k1)[:inds]
                        sol[i] = reindex(sol[i], length(all), ind)
                    end
                end

                if row == 1

                    objectives = conditional_probabs(upper_mpo, lower_mps, new_s, sol)

                else
                    # upper cutoff
                    ind_above = [0 for _ in 1:s[2]]
                    for k in 1:s[2]
                        l = grid[row-1,k]
                        l1 = grid[row,k]
                        all = props(gg, l)[:spins]

                        index = ps.spins[k+(row-2)*s[2]]
                        ind = props(gg, l, l1)[:inds]
                        ind_above[k] = reindex(index, length(all), ind)
                    end

                    if row < s[1]

                        upper_mps = [upper_mpo[k][:,:,ind_above[k],:,:] for k in 1:s[2]]
                        objectives = conditional_probabs(upper_mps, lower_mps, new_s, sol)

                    else
                        l = length(sol)
                        mps = [upper_mpo[k][:,:,ind_above[k],:] for k in l+1:s[2]]
                        mps = set_spin_from_letf(mps, l, new_s, s[2])
                        objectives = conditional_probabs(mps)
                    end
                end

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

    # TODO this will need to be corrected
    if false
        props(ns)[:grid]
        objectives = [sol.objective for sol in partial_s]
        spins = [sol.spins for sol in partial_s]
        spins = [map(i->ind2spin(i, 1)[1], sol) for sol in spins]
        return spins[end:-1:1], objectives[end:-1:1]
    end


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
