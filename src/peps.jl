using TensorOperations
using LinearAlgebra
using GenericLinearAlgebra



function compute_single_tensor(ns::Vector{Node_of_grid}, qubo::Vector{Qubo_el{T}},
                                                        i::Int, β::T) where T <: AbstractFloat

    n = ns[i]
    i == n.i || error("$i ≠ $(n.i), error in indexing a grid")

    tensor_size = map(x -> 2^length(x), [n.left, n.right, n.up, n.down, n.spin_inds])
    # for the initial mps

    tensor = zeros(T, (tensor_size...))

    if length(n.spin_inds) == 1
        j = n.spin_inds[1]

        h = JfromQubo_el(qubo, j,j)
        Jil = T(0.)
        if length(n.left) > 0
            Jil = JfromQubo_el(qubo, n.left[1]...)
        end

        Jiu = T(0.)
        if length(n.up) > 0
            Jiu = JfromQubo_el(qubo, n.up[1]...)
        end

        for k in CartesianIndices(tuple(tensor_size...))

            b = [ind2spin(k[i], tensor_size[i]) for i in 1:5]
            tensor[k] = Tgen(b..., Jil, Jiu, h, β)
        end
    end
    if length(n.down) == 0
        return dropdims(tensor, dims = 4)
    elseif length(n.up) == 0
        return dropdims(tensor, dims = 3)
    end
    return tensor
end

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


function MPOxMPO(mpo_down::Vector{Array{T, 4}}, mpo_up::Vector{Array{T, 4}}) where T <: AbstractFloat
        mpo_res = Array{Union{Nothing, Array{T}}}(nothing, length(mpo_down))
        for i in 1:length(mpo_down)
        A = mpo_down[i]
        B = mpo_up[i]
        sa = size(A)
        sb = size(B)

        C = zeros(T, sa[1] , sb[1], sa[2], sb[2], sb[3], sa[4])
        @tensor begin
            C[a,d,b,e,f,g] = A[a,b,x,g]*B[d,e,f,x]
        end
        mpo_res[i] = reshape(C, (sa[1]*sb[1], sa[2]*sb[2], sb[3], sa[4]))
    end
    Array{Array{T, 4}}(mpo_res)
end

"""
    function compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat

for general implementation
"""

function compute_scalar_prod(mps_down::Vector{Array{T, 3}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
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

function compute_scalar_prod(mps_down::Vector{Array{T, 3}}, mps_up::Vector{Array{T, N} where N}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    env[1,1,:]
end

function compute_scalar_prod(mps_down::Vector{Array{T, 3}}, mps_up::Vector{Array{T, 3}}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    env
end

function scalar_prod_step(mps_down::Array{T, 3}, mps_up::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1))

    @tensor begin
        C[a,b] = mps_up[a,x,z]*mps_down[b,y,z]*env[x,y]
    end
    C
end


function scalar_prod_step(mps_down::Array{T, 3}, mps_up::Array{T, 4}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1), size(mps_up, 4))

    @tensor begin
        C[a,b, u] = mps_up[a,x,z,u]*mps_down[b,y,z]*env[x,y]
    end
    C
end

function scalar_prod_step(mps_down::Array{T, 3}, mps_up::Array{T, 3}, env::Array{T, 3}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1), size(env, 3))

    @tensor begin
        C[a,b,u] = mps_up[a,x,z]*mps_down[b,y,z]*env[x,y,u]
    end
    C
end


function set_part(M::Vector{Array{T,3}}, s::Vector{Int} = Int[]) where T <: AbstractFloat
    l = length(s)
    siz = size(M,1)
    chain = [M[l+1]]
    chain = vcat(chain, [sum_over_last(M[i]) for i in (l+2):siz])

    if l > 0
        v = [T(s[l] == -1), T(s[l] == 1)]
        K1 = reshape(v, (1,2))
        chain = vcat([K1], chain)
    end

    Vector{Array{T, N} where N}(chain)
end

function conditional_probabs(M::Vector{Array{T,4}}, lower_mps::Vector{Array{T,3}}, s::Vector{Int} = Int[]) where T <: AbstractFloat

    l = length(s)
    siz = size(M,1)

    upper = [M[l+1]]

    if l < siz-1
        # this can be cashed
        upper = vcat(upper, [sum_over_last(M[i]) for i in (l+2):siz])

    end
    if l > 0
        v = [T(s[l] == -1), T(s[l] == 1)]
        K = reshape(kron(v, v), (1,2,2))
        upper = vcat([K], upper)

        for j in l-1:-1:1
            v = [T(s[j] == -1), T(s[j] == 1)]
            K1 = reshape(v, (1,1,2))
            upper = vcat([K1], upper)
        end
    end

    unnorm_prob = compute_scalar_prod(lower_mps, upper)
    unnorm_prob./sum(unnorm_prob)
end


function chain2point(chain::Vector{Array{T, N} where N}) where T <: AbstractFloat
    l = length(chain)

    env = ones(T, 1)
    for i in l:-1:1
        env = chain2pointstep(chain[i], env)
    end
    env[1,:]./sum(env)
end

function chain2pointstep(t::Array{T, 3}, env::Array{T, 1}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1), size(t, 3))

    @tensor begin
        ret[a,b] = t[a, x, b]*env[x]
    end
    ret
end

function chain2pointstep(t::Array{T, 2}, env::Array{T, 1}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1))

    @tensor begin
        ret[a] = t[a, x]*env[x]
    end
    ret
end

function chain2pointstep(t::Array{T,2}, env::Array{T,2}) where T <: AbstractFloat
    ret = zeros(T, size(t, 1), size(env, 2))

    @tensor begin
        ret[a,b] = t[a, x]*env[x, b]
    end
    ret
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

here the new objective is multipmed by the old one, hence we can take advantage
of the marginal probabilities
"""

function add_spin(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol{T}(vcat(ps.spins, [s]), ps.objective*objective)
end

function add_spin_marginal(ps::Partial_sol{T}, s::Int, objective::T) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol{T}(vcat(ps.spins, [s]), objective)
end


function solve(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T, χ::Int = 0, threshold::T = T(1e-14)) where T <: AbstractFloat
    problem_size = maximum(grid)
    ns = [Node_of_grid(i, grid) for i in 1:maximum(grid)]
    s = size(grid)
    #M = make_pepsTN(grid, qubo, β)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row in 1:s[1]

        #this may need to ge cashed
        lower_mps = make_lower_mps(grid, ns, qubo, row + 1, β, χ, threshold)

        for j in grid[row,:]

            partial_s_temp = Partial_sol{T}[Partial_sol{T}()]
            for ps in partial_s

                part_sol = ps.spins
                sol = part_sol[1+(row-1)*s[2]:end]

                objectives = [0., 0.]

                if row == 1

                    A = [compute_single_tensor(ns, qubo, j, β) for j in grid[row,:]]
                    objectives = conditional_probabs(A, lower_mps, sol)

                elseif row < s[1]
                    sol_row = part_sol[1+(row-2)*s[2]:(row-1)*s[2]]
                    A = [compute_single_tensor(ns, qubo, j, β) for j in grid[row,:]]
                    A = [A[j][:,:,spins2ind(sol_row[j]),:,:] for j in 1:length(sol_row)]

                    objectives = conditional_probabs(A, lower_mps, sol)
                else
                    sol_row = part_sol[1+(row-2)*s[2]:(row-1)*s[2]]
                    A = [compute_single_tensor(ns, qubo, j, β) for j in grid[row,:]]
                    A = [A[j][:,:,spins2ind(sol_row[j]),:] for j in 1:length(sol_row)]

                    chain = set_part(A, sol)
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
                # sort from the ground state
                return partial_s[end:-1:1]
            end
        end
    end
end


function make_left_canonical(t1::Array{T, 3}, t2::Array{T, 3}) where T <: AbstractFloat
    s = size(t1)

    p1 = [1,3,2]
    A1 = permutedims(t1, p1)
    A1 = reshape(A1, (s[1]*s[3], s[2]))

    U,Σ,V = svd(A1)
    T2 = diagm(Σ)*transpose(V)
    k = length(Σ)

    Anew = reshape(U, (s[1], s[3], k))
    Anew = permutedims(Anew, invperm(p1))

    @tensor begin
        t2[a,b,c] := T2[a,x]*t2[x,b,c]
    end
    Anew, t2
end



function make_right_canonical(t1::Array{T, 3}, t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)

    B2 = reshape(t2, (s[1], s[2]*s[3]))

    U,Σ,V = svd(B2)
    k = length(Σ)
    T1 = U*diagm(Σ)
    V = transpose(V)

    Bnew = reshape(V, (k, s[2], s[3]))

    @tensor begin
        t1[a,b,c] := T1[x,b]*t1[a,x,c]
    end

    t1, Bnew
end

function vec_of_right_canonical(mps::Vector{Array{T, 3}}) where T <: AbstractFloat
    for i in length(mps)-1:-1:1
        mps[i], mps[i+1] = make_right_canonical(mps[i], mps[i+1])
    end
    mps
end


function vec_of_left_canonical(mps::Vector{Array{T, 3}}) where T <: AbstractFloat
    for i in 1:length(mps)-1
        mps[i], mps[i+1] = make_left_canonical(mps[i], mps[i+1])
    end
    mps
end

function left_canonical_approx(mps::Vector{Array{T, 3}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_left_canonical(copy(mps))
    if χ == 0
        return mps
    else
        for i in 1:length(mps)-1
            s = size(mps[i], 2)
            χ1 = min(s, χ)

            mps[i] = mps[i][:,1:χ1,:]
            mps[i+1] = mps[i+1][1:χ1,:,:]
        end
    end
    mps
end

function right_canonical_approx(mps::Vector{Array{T, 3}}, χ::Int) where T <: AbstractFloat

    mps = vec_of_right_canonical(copy(mps))
    if χ == 0
        return mps
    else
        for i in 2:length(mps)
            s = size(mps[i], 1)
            χ1 = min(s, χ)

            mps[i-1] = mps[i-1][:,1:χ1,:]
            mps[i] = mps[i][1:χ1,:,:]
        end
    end
    mps
end


function QR_make_right_canonical(t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)
    p = [2,3,1]
    t2 = permutedims(t2, p)

    B2 = reshape(t2, (s[2]*s[3], s[1]))
    Q,R = qr(B2)
    Q = Q[:,1:size(R,1)]

    l = min(size(Q,2), s[1])

    Bnew = reshape(Q, (s[2], s[3], l))
    Bnew = permutedims(Bnew, invperm(p))

    Bnew, R
end

function QR_make_left_canonical(t2::Array{T, 3}) where T <: AbstractFloat

    s = size(t2)
    p = [1,3,2]
    t2 = permutedims(t2, p)

    B2 = reshape(t2, (s[1]*s[3], s[2]))
    Q,R = qr(B2)
    Q = Q[:,1:size(R,1)]
    l = min(size(Q,2), s[2])

    Bnew = reshape(Q, (s[1], s[3], l))
    Bnew = permutedims(Bnew, invperm(p))

    Bnew, R
end

function R_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(U, 1), size(U_exact, 1))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = U[a,x,v]*U_exact[b,y,v]*R[x,y]
    end
    C
end

function L_update(U::Array{T, 3}, U_exact::Array{T, 3}, R::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(U, 2), size(U_exact, 2))
    @tensor begin
        C[a,b] = U[x,a,v]*U_exact[y,b,v]*R[x,y]
    end
    C
end

function compress_mps_itterativelly(mps::Vector{Array{T,3}}, mps_anzatz::Vector{Array{T,3}}, χ::Int, threshold::T) where T <: AbstractFloat

    mps_centr = [zeros(T, 1,1,1) for _ in 1:length(mps)]

    mps_ret = [zeros(T, 1,1,1) for _ in 1:length(mps)]
    mps_ret1 = [zeros(T, 1,1,1) for _ in 1:length(mps)]
    maxsweeps = 300
    maxsweeps = 5

    # initialize R and L
    all_L = [ones(T,1,1) for _ in 1:length(mps)]
    for i in 1:length(mps)-1
        all_L[i+1] = L_update(mps_anzatz[i], mps[i], all_L[i])
    end
    all_R = [ones(T,1,1) for _ in 1:length(mps)]

    s = size(mps[end],2)
    R_exact = Matrix{T}(I, s,s)

    for sweep = 1:maxsweeps
        n = 0.
        ϵ = 0.
        for i in length(mps):-1:1
            # transform to canonical centre
            @tensor begin
                mps_c[a,b,c] := mps[i][a,x,c]*R_exact[b,x]
            end
            mps_centr[i] = mps_c

            @tensor begin
                #v concers contracting modes of size 1 in C
                M[a,b,c] := all_L[i][a,y]*mps_c[y,z,c]*all_R[i][b,z]
            end

            Q, TD = QR_make_right_canonical(M)
            Q_exact, R_exact = QR_make_right_canonical(mps_c)

            # compute ϵ
            @tensor begin
                X[x,y] := M[x,a,b]*M[y,a,b]
            end
            if n == 0.
                n = norm(R_exact)
            end
            ϵ = ϵ + 1-tr(X./n^2)

            mps_ret[i] = Q
            if i > 1
                all_R[i-1] = R_update(Q, Q_exact, all_R[i])
            end
        end
        if false
            println("ϵ l2r = ", ϵ)
        end

        ϵ = 0.

        s = size(mps[1],1)
        R_exact = Matrix{T}(I, s,s)
        for i in 1:length(mps)

            mps_c = mps_centr[i]

            @tensor begin
                #v concers contracting modes of size 1 in C
                M[a,b,c] := all_L[i][a,y]*mps_c[y,z,c]*all_R[i][b,z]
            end

            Q, TD = QR_make_left_canonical(M)
            mps_ret1[i] = Q

            @tensor begin
                X[x,y] := M[x,a,b]*M[y,a,b]
            end
            ϵ = ϵ + 1-tr(X./n^2)

            if i < length(mps)
                A = L_update(Q, mps[i], all_L[i])
                all_L[i+1] = A
            end
        end
        if false
            println("ϵ r2l = ", ϵ)
        end
        if abs(ϵ) < threshold
            return mps_ret1
        end
    end
    mps_ret1
end

# various compressing schemes

function compress_iter(mps::Vector{Array{T,3}}, χ::Int, threshold::T) where T <: AbstractFloat
    mps_lc = left_canonical_approx(mps, 0)
    mps_anzatz = left_canonical_approx(mps, χ)
    compress_mps_itterativelly(mps_lc, mps_anzatz, χ, threshold)
end


function compress_svd(mps::Vector{Array{T,3}}, χ::Int) where T <: AbstractFloat
    left_canonical_approx(mps, χ)
end
