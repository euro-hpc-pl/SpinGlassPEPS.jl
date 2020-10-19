using TensorOperations
using LinearAlgebra
using GenericLinearAlgebra


"""
compute_single_tensor(ns::Vector{Node_of_grid}, qubo::Vector{Qubo_el{T}}, i::Int, β::T)

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

f(i::Int) = div((i-1), 2)+1


function compute_single_tensor(ns::Vector{Node_of_grid}, qubo::Vector{Qubo_el{T}},
                                                        i::Int, β::T) where T <: AbstractFloat


    n = ns[i]
    i == n.i || error("$i ≠ $(n.i), error in indexing a grid")

    tensor_size = map(x -> 2^length(x), [n.left, n.right, n.up, n.down, n.spin_inds])

    tensor = zeros(T, (tensor_size...))


    Jil = T[]
    Jiu = T[]
    h = T[]

    left = Int[]
    right = Int[]
    up = Int[]
    down = Int[]

    for j in n.spin_inds

        push!(h, JfromQubo_el(qubo, j,j))

        if j in [e[1] for e in n.left]
            #this [1] has to be changed for pegasus
            a = findall(x->x[1]==j, n.left)[1]
            push!(Jil, JfromQubo_el(qubo, n.left[a]...))

            b = findall(x->x[1]==j, n.spin_inds)[1]
            push!(left, b)
        end

        if j in [e[1] for e in n.right]
            a = findall(x->x[1]==j, n.spin_inds)[1]
            push!(right, a)
        end

        if j in [e[1] for e in n.down]
            a = findall(x->x[1]==j, n.spin_inds)[1]
            push!(down, a)
        end

        if j in [e[1] for e in n.up]
            a = findall(x->x[1]==j, n.up)[1]
            push!(Jiu, JfromQubo_el(qubo, n.up[a]...))
            b = findall(x->x[1]==j, n.spin_inds)[1]
            push!(up, b)
        end
    end

    for k in CartesianIndices(tuple(tensor_size...))

        spins = [ind2spin(k[i], tensor_size[i]) for i in 1:5]

        J1 = 0.
        if length(Jil) > 0
            J1 = sum(Jil.*spins[1].*spins[5][left])
        end

        J2 = 0.
        if length(Jiu) > 0
            J2 = sum(Jiu.*spins[3].*spins[5][up])
        end

        hh = β*sum(h.*spins[5])

        if n.intra_struct != Array{Int64,1}[]
            for pair in n.intra_struct
                a = findall(x->x==pair[1], n.spin_inds)[1]
                b = findall(x->x==pair[2], n.spin_inds)[1]

                s1 = spins[5][a]
                s2 = spins[5][b]
                J = JfromQubo_el(qubo, pair[1], pair[2])
                #println(2*β*J*s1*s2)
                hh = hh + 2*β*J*s1*s2
            end
        end

        r = 1.
        if length(right) > 0
            r = T(spins[2] == spins[5][right])
        end

        d = 1.
        if length(down) > 0
            d = T(spins[4] == spins[5][down])
        end

        tensor[k] = r*d*exp(β*2*J1)*exp(β*2*J2)*exp(hh)
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
function set_spin_from_letf(mps::Vector{Array{T,3}}, sol::Vector{Int}, s::Int) where T <: AbstractFloat

    Ms = []
    # no element to the left
    if length(sol) == 0
        # transposition is to make a simple matrix multiplication for computing a probability
        Ms = [Array(transpose(mps[1][1,:,:]))]
    else
        Ms = [Array(transpose(mps[1][sol[end],:,:]))]
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
function set_spin_from_letf(mpo::Vector{Array{T,4}}, sol::Vector{Int}, s::Int) where T <: AbstractFloat
    l = length(sol)
    B = []
    if l == 0
        B = [mpo[1][1,:,:,:]]
    else
        B = [mpo[1][sol[end],:,:,:]]
    end
    if l < s
        B = vcat(B, [sum_over_last(el) for el in mpo[2:end]])
    end
    return B
end


function conditional_probabs(M::Vector{Array{T,4}}, lower_mps::Vector{Array{T,3}}, s::Vector{Int} = Int[]) where T <: AbstractFloat

    l1 = length(lower_mps)
    l = length(s)

    A = set_spin_from_letf(M[l+1:end], s, l1-1)

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


function conditional_probabs(chain::Vector{Array{T, 2}}) where T <: AbstractFloat
    env = ones(T, 1)
    for i in length(chain):-1:1
        env = chain[i]*env
    end
    env./sum(env)
end


function make_lower_mps(grid::Matrix{Int}, ns::Vector{Node_of_grid},
                                           qubo::Vector{Qubo_el{T}}, k::Int, β::T, χ::Int, threshold::T) where T <: AbstractFloat
    s = size(grid,1)
    if k <= s
        mps = [sum_over_last(compute_single_tensor(ns, qubo, j, β)) for j in grid[s,:]]
        for i in s-1:-1:k

            mpo = [sum_over_last(compute_single_tensor(ns, qubo, j, β)) for j in grid[i,:]]
            mps = MPSxMPO(mps, mpo)
        end
        if threshold == 0.
            return mps
        end
        return compress_iter(mps, χ, threshold)
        #return compress_svd(mps, χ)
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
    add_spin(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat

Add a spin and replace an objective function to Partial_sol{T} type
"""

function add_spin(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat
    Partial_sol{T}(vcat(ps.spins, [s]), objective)
end


function solve(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T, χ::Int = 0, threshold::T = T(1e-14)) where T <: AbstractFloat
    problem_size = maximum(grid)

    # this will be moved outside
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]

    physical_dim = 2
    s = size(grid)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:s[1]

        #this may need to ge cashed
        lower_mps = make_lower_mps(grid, ns, qubo, row + 1, β, χ, threshold)

        upper_mpo = [compute_single_tensor(ns, qubo, j, β) for j in grid[row,:]]

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[]
            for ps in partial_s
                sol = ps.spins[1+(row-1)*s[2]:end]

                objectives = [T(0.) for _ in 1:physical_dim]

                if row == 1
                    objectives = conditional_probabs(upper_mpo, lower_mps, sol)

                elseif row < s[1]
                    # set spins from above
                    upper_mps = [upper_mpo[j][:,:,ps.spins[j+(row-2)*s[2]],:,:] for j in 1:s[2]]
                    objectives = conditional_probabs(upper_mps, lower_mps, sol)

                else
                    l = length(sol)
                    mps = [upper_mpo[j][:,:,ps.spins[j+(row-2)*s[2]],:] for j in l+1:s[2]]

                    chain = set_spin_from_letf(mps, sol, s[2])
                    objectives = conditional_probabs(chain)
                end

                for l in 1:physical_dim
                    push!(partial_s_temp, add_spin(ps, l, ps.objective*objectives[l]))
                end

            end
            partial_s = select_best_solutions(partial_s_temp, no_sols)

            if j == problem_size
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
            ii = ind2spin(ses[k])
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
