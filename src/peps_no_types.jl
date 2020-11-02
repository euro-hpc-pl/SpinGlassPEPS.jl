using TensorOperations
using LinearAlgebra
using GenericLinearAlgebra
using SharedArrays
using Distributed

# TODO β and interactions should be as Float64, if typechange, make it inside a solver

"""
compute_single_tensor(ns::Vector{Node_of_grid}, interactions::Vector{Interaction{T}}, i::Int, β::T)

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

# TODO this function needs to be clearer (split into a few) and perhaps spead up
# many repeating code inside

#TODO searching functions outside

function index_of_interacting_spins(n::Node_of_grid, pairs::Vector{Vector{Int}})
    spins = Int[]

    for el in pairs
        index = findall(x->x==el[1], n.spin_inds)[1]
        push!(spins, index)
    end
    spins
end

function compute_single_tensor(ns::Vector{Node_of_grid}, interactions::Vector{Interaction{T}},
                                                        i::Int, β::T) where T <: AbstractFloat


    n = ns[i]
    i == n.i || error("$i ≠ $(n.i), error in indexing a grid")

    tensor_size = map(x -> 2^length(x), [n.left, n.right, n.up, n.down, n.spin_inds])

    # these 2 are used to compute external energy
    Jil = [getJ(interactions, l...) for l in n.left]
    Jiu = [getJ(interactions, u...) for u in n.up]

    # these 2 are used to compute internal energy
    # compute log - energy here
    J_intra = [getJ(interactions, pair[1], pair[2]) for pair in n.intra_struct]
    h = [getJ(interactions, j,j) for j in n.spin_inds]

    left = index_of_interacting_spins(n, n.left)
    right = index_of_interacting_spins(n, n.right)
    up = index_of_interacting_spins(n, n.up)
    down = index_of_interacting_spins(n, n.down)

    tensor = zeros(T, (tensor_size...))
    # following are done outside the loop
    siz = [ceil(Int, log(2, size)) for size in tensor_size]

    # this will go to compute internal energy
    ind_a = Int[]
    ind_b = Int[]
    for pair in n.intra_struct
        push!(ind_a, findall(x->x==pair[1], n.spin_inds)[1])
        push!(ind_b, findall(x->x==pair[2], n.spin_inds)[1])
    end

    for k in CartesianIndices(tuple(tensor_size...))

        spins = [ind2spin(k[i], siz[i]) for i in 1:5]
        all_spins = spins[5]

        # dirac delta implementation
        r = true
        if length(right) > 0
            r = (spins[2] == all_spins[right])
        end

        d = true
        if length(down) > 0
            d = (spins[4] == all_spins[down])
        end
        # if any is false further is not necessary
        if (d && r)

            J1 = 0.
            if length(Jil) > 0
                J1 = sum(Jil.*spins[1].*all_spins[left])
            end

            J2 = 0.
            if length(Jiu) > 0
                J2 = sum(Jiu.*spins[3].*all_spins[up])
            end

            # this should be a function compute internal energy
            hh = β*sum(h.*all_spins)

            if n.intra_struct != Array{Int64,1}[]
                i = 1
                for pair in n.intra_struct

                    s1 = all_spins[ind_a[i]]
                    s2 = all_spins[ind_b[i]]

                    hh = hh + 2*β*J_intra[i]*s1*s2
                    i = i + 1
                end
            end

            @inbounds tensor[k] = exp(β*2*(J1+J2)+hh)
        end
    end

    if length(n.down) == 0
        return dropdims(tensor, dims = 4)
    elseif length(n.up) == 0
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
function set_spin_from_letf(mps::Vector{Array{T,3}}, sol::Vector{Int}, new_s::Int, s::Int) where T <: AbstractFloat

    Ms = []
    # no element to the left
    if length(sol) == 0
        # transposition is to make a simple matrix multiplication for computing a probability
        Ms = [Array(transpose(mps[1][1,:,:]))]
    else
        Ms = [Array(transpose(mps[1][new_s,:,:]))]
    end
    # otherwise no elements to the right
    if length(sol) < s
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
function set_spin_from_letf(mpo::Vector{Array{T,4}}, sol::Vector{Int}, new_s::Int, s::Int) where T <: AbstractFloat
    l = length(sol)
    B = []
    if l == 0
        B = [mpo[1][1,:,:,:]]
    else
        B = [mpo[1][new_s,:,:,:]]
    end
    if l < s
        B = vcat(B, [sum_over_last(el) for el in mpo[2:end]])
    end
    return B
end

# TODO, rename and explain s, s_new it was done temporarly
# explain tensors B,C,D
function conditional_probabs(M::Vector{Array{T,4}}, lower_mps::Vector{Array{T,3}}, new_s::Int, s::Vector{Int} = Int[]) where T <: AbstractFloat

    l1 = length(lower_mps)
    l = length(s)

    A = set_spin_from_letf(M[l+1:end], s, new_s, l1-1)

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


function make_lower_mps(grid::Matrix{Int}, ns::Vector{Node_of_grid},
                                           interactions::Vector{Interaction{T}}, k::Int, β::T, χ::Int, threshold::T) where T <: AbstractFloat
    s = size(grid,1)
    if k <= s
        mps = [sum_over_last(compute_single_tensor(ns, interactions, j, β)) for j in grid[s,:]]
        if threshold > 0.
            mps = compress_iter(mps, χ, threshold)
        end
        for i in s-1:-1:k

            mpo = [sum_over_last(compute_single_tensor(ns, interactions, j, β)) for j in grid[i,:]]
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

function solve(interactions::Vector{Interaction{T}}, ns::Vector{Node_of_grid}, grid::Matrix{Int},
                                        no_sols::Int = 2; β::T, χ::Int = 0,
                                        threshold::T = T(1e-14)) where T <: AbstractFloat

    s = size(grid)
    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:s[1]
        println("row of peps = ", row)
        #this may need to ge cashed
        lower_mps = make_lower_mps(grid, ns, interactions, row + 1, β, χ, threshold)

        upper_mpo = [compute_single_tensor(ns, interactions, j, β) for j in grid[row,:]]

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[]
            for ps in partial_s
                sol = ps.spins[1+(row-1)*s[2]:end]

                objectives = [T(0.)]
                new_s = 0
                if length(ns[j].left) > 0
                    k = findall(x->x[1]==j, grid[row,:])[1]
                    l = grid[row,:][k-1]
                    all = ns[l].spin_inds
                    conecting = [e[2] for e in ns[j].left]

                    new_s = reindex(sol[end], all, conecting)
                end

                if  row < s[1]
                    for i in 1:length(sol)
                        k = grid[row,i]
                        all = ns[k].spin_inds
                        conecting = [e[1] for e in ns[k].down]
                        sol[i] = reindex(sol[i], all, conecting)

                    end
                end

                if row == 1

                    objectives = conditional_probabs(upper_mpo, lower_mps, new_s, sol)

                else
                    ind_above = [0 for _ in 1:s[2]]
                    for k in 1:s[2]
                        l = grid[row-1,k]
                        all = ns[l].spin_inds
                        conecting = [e[1] for e in ns[l].down]

                        index = ps.spins[k+(row-2)*s[2]]
                        index = reindex(index, all, conecting)
                        ind_above[k] = index
                    end

                    if row < s[1]

                    # set spins from above
                        upper_mps = [upper_mpo[k][:,:,ind_above[k],:,:] for k in 1:s[2]]
                        objectives = conditional_probabs(upper_mps, lower_mps, new_s, sol)

                    else
                        l = length(sol)
                        mps = [upper_mpo[k][:,:,ind_above[k],:] for k in l+1:s[2]]


                        mps = set_spin_from_letf(mps, sol, new_s, s[2])
                        objectives = conditional_probabs(mps)
                    end
                end

                for l in 1:length(objectives)
                    push!(partial_s_temp, update_partial_solution(ps, l, ps.objective*objectives[l]))
                end

            end
            partial_s = select_best_solutions(partial_s_temp, no_sols)

            if j == maximum(grid)
                return return_solutions(partial_s, ns)
            end
        end
    end
end

"""
    return_solutions(partial_s::Vector{Partial_sol{T}})

return final solutions sorted backwards in form Vector{Partial_sol{T}}
spins are given in -1,1
"""
function return_solutions(partial_s::Vector{Partial_sol{T}}, ns::Vector{Node_of_grid})  where T <: AbstractFloat

    l = length(partial_s)
    objective = zeros(T, l)
    spins = [Int[] for _ in 1:l]
    size = get_system_size(ns)
    # order is reversed, to correspond with sort
    for i in 1:l
        one_solution = zeros(Int, size)
        objective[l-i+1] = partial_s[i].objective

        ses = partial_s[i].spins

        for k in 1:length(ns)

            ii = ind2spin(ses[k], length(ns[k].spin_inds))
            for j in 1:length(ii)
                one_solution[ns[k].spin_inds[j]] = ii[j]
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
