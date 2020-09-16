using TensorOperations

include("notation.jl")

# axiliary

function JfromQubo_el(qubo::Vector{Qubo_el{T}}, i::Int, j::Int) where T <: AbstractFloat
    try
        return filter(x->x.ind==(i,j), qubo)[1].coupling
    catch
        return filter(x->x.ind==(j,i), qubo)[1].coupling
    end
end


function make_tensor_sizes(l::Bool, r::Bool, u::Bool, d::Bool, s_virt::Int = 2, s_phys::Int = 2)
    tensor_size = [1,1,1,1,s_phys]
    if l
        tensor_size[1] = s_virt
    end
    if r
        tensor_size[2] = s_virt
    end
    if u
        tensor_size[3] = s_virt
    end
    if d
        tensor_size[4] = s_virt
    end
    (tensor_size..., )
end

function make_peps_node(grid::Matrix{Int}, qubo::Vector{Qubo_el{T}}, i::Int, β::T) where T <: AbstractFloat

    ind = findall(x->x==i, grid)[1]
    h = filter(x->x.ind==(i,i), qubo)[1].coupling
    bonds = [[0], [0], [0], [0], [-1,1]]

    # determine bonds directions from grid
    l = 0 < ind[2]-1
    r = ind[2]+1 <= size(grid, 2)
    u = 0 < ind[1]-1
    d = ind[1]+1 <= size(grid, 1)

    if r
        bonds[2] = [-1,1]
    end

    Jil = T(0.)
    if l
        j = grid[ind[1], ind[2]-1]
        Jil = JfromQubo_el(qubo, i,j)
        bonds[1] = [-1,1]
    end

    if d
        bonds[4] = [-1,1]
    end

    Jiu = T(0.)
    if u
        j = grid[ind[1]-1, ind[2]]
        Jiu = JfromQubo_el(qubo, i,j)
        bonds[3] = [-1,1]
    end

    tensor_size = make_tensor_sizes(l,r,u,d,2,2)
    tensor = zeros(T, tensor_size)

    for i in CartesianIndices(tensor_size)
        b = [bonds[j][i[j]] for j in 1:5]
        tensor[i] = Tgen(b..., Jil, Jiu, h, β)
    end
    tensor
end

# tensor network

function make_pepsTN(grid::Matrix{Int}, qubo::Vector{Qubo_el{T}}, β::T) where T <: AbstractFloat
    s = size(grid)
    M_of_tens = Array{Union{Nothing, Array{T}}}(nothing, s)
    for i in 1:prod(s)
        ind = findall(x->x==i, grid)[1]
        M_of_tens[ind] = make_peps_node(grid, qubo, i, β)
    end
    Array{Array{T, 5}}(M_of_tens)
end


function trace_all_spins(mps::Vector{Array{T, 5}}) where T <: AbstractFloat
    l = length(mps)
    traced_mps = Array{Union{Nothing, Array{T, 4}}}(nothing, l)
    for i in 1:l
        traced_mps[i] = sum_over_last(mps[i])
    end
    Vector{Array{T, 4}}(traced_mps)
end

function MPSxMPO(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
        mps_res = Array{Union{Nothing, Array{T}}}(nothing, length(mps_down))
        for i in 1:length(mps_down)
        A = mps_down[i]
        B = mps_up[i]
        sa = size(A)
        sb = size(B)

        C = zeros(T, sa[1] , sb[1], sa[2], sb[2], sb[3], sa[4])
        @tensor begin
            C[a,d,b,e,f,c] = A[a,b,x,c]*B[d,e,f,x]
        end
        mps_res[i] = reshape(C, (sa[1]*sb[1], sa[2]*sb[2], sb[3], sa[4]))
    end
    Array{Array{T, 4}}(mps_res)
end

function MPSxMPO(mps_down::Vector{Array{T, 5}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
        mps_res = Array{Union{Nothing, Array{T}}}(nothing, length(mps_down))
        for i in 1:length(mps_down)
        A = mps_down[i]
        B = mps_up[i]
        sa = size(A)
        sb = size(B)

        C = zeros(T, sa[1] , sb[1], sa[2], sb[2], sb[3], sa[4], sa[5])
        @tensor begin
            C[a,d,b,e,f,c, g] = A[a,b,x,c,g]*B[d,e,f,x]
        end
        mps_res[i] = reshape(C, (sa[1]*sb[1], sa[2]*sb[2], sb[3], sa[4], sa[5]))
    end
    Array{Array{T, 5}}(mps_res)
end

"""
    function compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat

for general implementation
"""

function compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    size(env) == (1,1) || error("output size $(size(env)) ≠ (1,1) not fully contracted")
    env[1,1]
end

"""
    compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, N} where N}) where T <: AbstractFloat

for peps implementation
"""

function compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, N} where N}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    env[1,1,:]
end

function scalar_prod_step(mps_down::Array{T, 4}, mps_up::Array{T, 4}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = mps_up[a,x,v,z]*mps_down[b,y,z,v]*env[x,y]
    end
    C
end


function scalar_prod_step(mps_down::Array{T, 4}, mps_up::Array{T, 5}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1), size(mps_up, 5))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b, u] = mps_up[a,x,v,z, u]*mps_down[b,y,z,v]*env[x,y]
    end
    C
end

function scalar_prod_step(mps_down::Array{T, 4}, mps_up::Array{T, 4}, env::Array{T, 3}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1), size(env, 3))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b, u] = mps_up[a,x,v,z]*mps_down[b,y,z,v]*env[x,y,u]
    end
    C
end


function set_part(M::Vector{Array{T,5}}, s::Vector{Int} = Int[]) where T <: AbstractFloat
    l = length(s)
    siz = size(M,1)
    chain = [M[l+1]]
    chain = vcat(chain, [sum_over_last(M[i]) for i in (l+2):siz])

    if l > 0
        v = [T(s[l] == -1), T(s[l] == 1)]
        K1 = reshape(v, (1,2,1,1))
        chain = vcat([K1], chain)
    end

    Vector{Array{T, N} where N}(chain)
end

function conditional_probabs(M::Vector{Array{T,5}}, lower_mps::Vector{Array{T,4}}, s::Vector{Int} = Int[]) where T <: AbstractFloat

    l = length(s)
    siz = size(M,1)

    upper = [M[l+1]]
    if l < siz-1
        # this can be cashed
        upper = vcat(upper, [sum_over_last(M[i]) for i in (l+2):siz])

    end
    if l > 0
        v = [T(s[l] == -1), T(s[l] == 1)]
        K = reshape(kron(v, v), (1,2,1,2))
        upper = vcat([K], upper)

        for j in l-1:-1:1
            v = [T(s[j] == -1), T(s[j] == 1)]
            K1 = reshape(v, (1,1,1,2))
            upper = vcat([K1], upper)
        end
    end

    unnorm_prob = compute_scalar_prod(lower_mps, upper)
    unnorm_prob./sum(unnorm_prob)
end


function set_row(mps::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    l = length(ses)
    upper = [ones(T, 1,1,1,1) for _ in 1:l]
    for i in 1:l
        v = [T(ses[i] == -1), T(ses[i] == 1)]
        upper[i] = reshape(v, (1,1,1,2))
    end

        MPSxMPO(mps, Vector{Array{T, 4}}(upper))
end

function chain2point(chain::Vector{Array{T, N} where N}) where T <: AbstractFloat
    l = length(chain)

    env = ones(T, 1)
    for i in l:-1:1
        env = chain2pointstep(chain[i], env)
    end
    env[1,:]./sum(env)
end

function chain2pointstep(t::Array{T, 5}, env::Array{T, 1}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1), size(t, 5))
    t = t[:,:,1,1,:]

    @tensor begin
        ret[a,b] = t[a, x, b]*env[x]
    end
    ret
end

function chain2pointstep(t::Array{T, 4}, env::Array{T, 1}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1))
    t = t[:,:,1,1]
    @tensor begin
        ret[a] = t[a, x]*env[x]
    end
    ret
end

function chain2pointstep(t::Array{T,4}, env::Array{T,2}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1), size(env, 2))
    t = t[:,:,1,1]

    @tensor begin
        ret[a,b] = t[a, x]*env[x, b]
    end
    ret
end


function make_lower_mps(M::Matrix{Array{T, 5}}, k::Int, χ::Int) where T <: AbstractFloat
    s = size(M,1)
    if k <= s
        mps = trace_all_spins(M[s,:])
        for i in s-1:-1:k
            mpo = trace_all_spins(M[i,:])
            mps = MPSxMPO(mps, mpo)
        end
        return left_canonical_approx(mps, χ)
    end
    0
end


mutable struct Partial_sol{T <: AbstractFloat}
    spins::Vector{Int}
    objective::T
    upper_mps::Vector{Array{T, 4}}
    function(::Type{Partial_sol{T}})(spins::Vector{Int}, objective::T, upper_mps::Vector{Array{T, 4}}) where T <:AbstractFloat
        new{T}(spins, objective, upper_mps)
    end
    function(::Type{Partial_sol{T}})(spins::Vector{Int}, objective::T) where T <:AbstractFloat
        new{T}(spins, objective, [zeros(0,0,0,0)])
    end
    function(::Type{Partial_sol{T}})() where T <:AbstractFloat
        new{T}(Int[], 1., [zeros(0,0,0,0)])
    end
end


function add_spin(ps::Partial_sol{T}, s::Int) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol{T}(vcat(ps.spins, [s]), T(0.), ps.upper_mps)
end

"""
    add_spin(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat

here the new objective is multipmed by the old one, hence we can take advantage
of the marginal probabilities
"""

function add_spin(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol{T}(vcat(ps.spins, [s]), ps.objective*objective, ps.upper_mps)
end


function solve(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T, χ::Int = 0) where T <: AbstractFloat
    problem_size = maximum(grid)
    s = size(grid)
    M = make_pepsTN(grid, qubo, β)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:s[1]

        #this may need to ge cashed
        lower_mps = make_lower_mps(M, row + 1, χ)

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[Partial_sol{T}()]
            for ps in partial_s

                part_sol = ps.spins
                sol = part_sol[1+(row-1)*s[2]:end]

                objectives = [0., 0.]

                if row == 1
                    objectives = conditional_probabs(M[row,:], lower_mps, sol)

                elseif row < s[1]
                    sol_row = part_sol[1+(row-2)*s[2]:(row-1)*s[2]]
                    Mtemp = set_row(M[row,:], sol_row)

                    objectives = conditional_probabs(Mtemp, lower_mps, sol)
                else
                    sol_row = part_sol[1+(row-2)*s[2]:(row-1)*s[2]]
                    Mtemp = set_row(M[row,:], sol_row)
                    chain = set_part(Mtemp, sol)
                    objectives = chain2point(chain)
                end

                a = add_spin(ps, -1, objectives[1])
                b = add_spin(ps, 1, objectives[2])

                if partial_s_temp[1].spins == []
                    partial_s_temp = vcat(a,b)
                else
                    partial_s_temp = vcat(partial_s_temp, a,b)
                end
            end

            obj = [ps.objective for ps in partial_s_temp]
            perm = sortperm(obj)
            p = last_m_els(perm, no_sols)
            partial_s = partial_s_temp[p]

            if j == problem_size
                    return partial_s
            end
        end
    end
end


function make_left_canonical(t1::Array{T, 4}, t2::Array{T, 4}) where T <: AbstractFloat
    s = size(t1)

    p1 = [1,3,4,2]
    A1 = permutedims(t1, p1)
    A1 = reshape(A1, (s[1]*s[3]*s[4], s[2]))

    U,Σ,V = svd(Float64.(A1))
    T2 = diagm(Σ)*transpose(V)
    k = length(Σ)

    Anew = reshape(U, (s[1], s[3], s[4], k))
    Anew = permutedims(Anew, invperm(p1))

    @tensor begin
        t2[a,b,c,d] := T2[a,x]*t2[x,b,c,d]
    end
    Anew, t2
end



function make_right_canonical(t1::Array{T, 4}, t2::Array{T, 4}) where T <: AbstractFloat

    s = size(t2)

    B2 = reshape(t2, (s[1], s[2]*s[3]*s[4]))

    U,Σ,V = svd(Float64.(B2))
    k = length(Σ)
    T1 = U*diagm(Σ)
    V = transpose(V)

    Bnew = reshape(V, (k, s[2], s[3], s[4]))

    @tensor begin
        t1[a,b,c,d] := T1[x,b]*t1[a,x,c,d]
    end

    t1, Bnew
end

function vec_of_right_canonical(mps::Vector{Array{T, 4}}) where T <: AbstractFloat
    for i in length(mps)-1:-1:1
        mps[i], mps[i+1] = make_right_canonical(mps[i], mps[i+1])
    end
    mps
end


function vec_of_left_canonical(mps::Vector{Array{T, 4}}) where T <: AbstractFloat
    for i in 1:length(mps)-1
        mps[i], mps[i+1] = make_left_canonical(mps[i], mps[i+1])
    end
    mps
end

function left_canonical_approx(mps::Vector{Array{T, 4}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_left_canonical(mps)
    if χ == 0
        return mps
    else
        for i in 1:length(mps)-1
            s = size(mps[i], 2)
            χ1 = min(s, χ)

            mps[i] = mps[i][:,1:χ1,:,:]
            mps[i+1] = mps[i+1][1:χ1,:,:,:]
        end
    end
    mps
end

function right_canonical_approx(mps::Vector{Array{T, 4}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_right_canonical(mps)
    if χ == 0
        return mps
    else
        for i in 2:length(mps)
            s = size(mps[i], 1)
            χ1 = min(s, χ)

            mps[i-1] = mps[i-1][:,1:χ1,:,:]
            mps[i] = mps[i][1:χ1,:,:,:]
        end
    end
    mps
end

function L_step(mps_anzatz::Array{T, 4}, mps::Array{T, 4}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_anzatz, 2), size(mps, 2))

    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = env[x,y]*mps_anzatz[x,a,v,z]*mps[y,b,v,z]
    end
    C
end


function R_update(mps_sol::Array{T, 4}, mps::Array{T, 4}, R::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_sol, 1), size(mps, 1))

    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = R[x,y]*mps_sol[x,a,v,z]*mps[y,b,v,z]
    end
    C
end

function compress_mps_itterativelly(mps::Vector{Array{T,4}}, χ::Int) where T <: AbstractFloat
    mps = left_canonical_approx(mps, 0)
    mps_anzatz = left_canonical_approx(mps, χ)

    all_L = [ones(T,1,1)]
    for i in 1:length(mps)-1
        E = L_step(mps_anzatz[i], mps[i], all_L[i])
        push!(all_L, E)
    end

    Tens = all_L[end]
    R = ones(T,1,1)
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b,c,d] := Tens[a,y]*mps[end][y,z,c,d]*R[b,z]
    end
    #println(C)
    #println(mps_anzatz[end])

        # mixed representation
        # itterative loop (minimisation of ε)
            # conctract mps exact with mps anzatz without the ith M
            # result of contraction -> M
            # ε = 1- ∑ trace(M' * M)
    mps_anzatz
end


function make_qubo()
    css = -2.
    qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
    qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end
train_qubo = make_qubo()


grid = [1 2 3; 4 5 6; 7 8 9]

ses = solve(train_qubo, grid, 1; β = 1.)

#ses1 = solve_arbitrary_decomposition(train_qubo, grid, 4; β = 1.)

for el in ses
    println(el.spins)
end

######## arbitrary decomposition  ############


function set_spins_on_mps(mps::Vector{Array{T, 5}}, s::Vector{Int}) where T <: AbstractFloat
    l = length(mps)
    up_bonds = zeros(Int, l)
    output_mps = Array{Union{Nothing, Array{T, 4}}}(nothing, l)
    for i in 1:l
        if s[i] == 0
            output_mps[i] = sum_over_last(mps[i])
        else
            A = set_last(mps[i], s[i])
            ind = spins2index(s[i])

            output_mps[i] = A

        end
    end
    Vector{Array{T, 4}}(output_mps)
end

function comp_marg_p(mps_u::Vector{Array{T, 4}}, mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)

    mps_n = MPSxMPO(mpo, mps_u)
    compute_scalar_prod(mps_d, mps_n), mps_n
end

function comp_marg_p_first(mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mps_u = set_spins_on_mps(M, ses)
    compute_scalar_prod(mps_d, mps_u), mps_u
end

function comp_marg_p_last(mps_u::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)

    compute_scalar_prod(mpo, mps_u)
end

function solve_arbitrary_decomposition(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T, χ = 0) where T <: AbstractFloat
    problem_size = maximum(grid)
    s = size(grid)
    M = make_pepsTN(grid, qubo, β)

    partial_solutions = Partial_sol{T}[Partial_sol{T}()]

    for row in 1:s[1]
        #this may need to ge cashed
        lower_mps = make_lower_mps(M, row + 1, χ)

        for j in grid[row,:]

            a = [add_spin(ps, 1) for ps in partial_solutions]
            b = [add_spin(ps, -1) for ps in partial_solutions]
            partial_solutions = vcat(a,b)

            for ps in partial_solutions

                part_sol = ps.spins
                sol = part_sol[1+(row-1)*s[2]:end]
                l = s[2] - length(sol)
                sol = vcat(sol, fill(0, l))

                u = []
                prob = 0.
                if row == 1
                    prob, u = comp_marg_p_first(lower_mps, M[row,:], sol)
                elseif row == s[1]
                    prob = comp_marg_p_last(ps.upper_mps, M[row,:], sol)
                else
                    prob, u = comp_marg_p(ps.upper_mps, lower_mps, M[row,:], sol)
                end

                ps.objective = prob

                if (j % s[2] == 0) & (j < problem_size)
                    ps.upper_mps = u
                end
            end

            objectives = [ps.objective for ps in partial_solutions]

            perm = sortperm(objectives)

            p1 = last_m_els(perm, no_sols)

            partial_solutions = partial_solutions[p1]

            if j == problem_size
                    return partial_solutions
            end
        end
    end
end

ses1 = solve_arbitrary_decomposition(train_qubo, grid, 1; β = 1.)

for el in ses1
    println(el.spins)
end
