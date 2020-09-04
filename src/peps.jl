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


function compute_scalar_prod(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, 4}}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    size(env) == (1,1) || error("output size $(size(env)) ≠ (1,1) not fully contracted")
    env[1,1]
end

function scalar_prod_step(mps_down::Array{T, 4}, mps_up::Array{T, 4}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b] = mps_up[a,x,v,z]*mps_down[b,y,z,v]*env[x,y]
    end
    C
end


function scalar_prod_step1(mps_down::Array{T, 4}, mps_up::Array{T, 5}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps_up, 1), size(mps_down, 1), size(mps_up, 5))
    @tensor begin
        #v concers contracting modes of size 1 in C
        C[a,b, u] = mps_up[a,x,v,z, u]*mps_down[b,y,z,v]*env[x,y]
    end
    C
end



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
            #if i > 1
            if false
                # breaks bonds between subsequent tensors in row
                # if s is set, excludes first element of the row
                output_mps[i] = A[ind:ind,:,:,:]
                output_mps[i-1] = output_mps[i-1][:,ind:ind,:,:]
            else
                output_mps[i] = A
            end
            if false
                size(A, 3) > 1
                up_bonds[i] = ind
                output_mps[i] = output_mps[i][:,:,ind:ind,:]
            end
        end
    end
    Vector{Array{T, 4}}(output_mps)
end

function reduce_bonds_horizontally!(mps::Vector{Array{T, 4}}, ses::Vector{Int}) where T <: AbstractFloat
    for i in 1:length(ses)
        j = ses[i]
        # if j is zero or dim has already been reduced
        # it will be skipped
        try
            mps[i] = mps[i][:,:,:,j:j]
        catch
            0
        end
    end
end

function comp_marg_p(mps_u::Vector{Array{T, 4}}, mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)
    #reduce_bonds_horizontally!(mps_u, s)
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
    #reduce_bonds_horizontally!(mps_u, s)
    compute_scalar_prod(mpo, mps_u)
end

function make_lower_mps(M::Matrix{Array{T, 5}}, k::Int, threshold::T) where T <: AbstractFloat
    s = size(M,1)
    if k <= s
        mps = trace_all_spins(M[s,:])
        for i in s-1:-1:k
            mpo = trace_all_spins(M[i,:])
            mps = MPSxMPO(mps, mpo)
        end
        if threshold != 0.
            mps = svd_approx(mps, threshold)
        end
        return mps
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
    function(::Type{Partial_sol{T}})() where T <:AbstractFloat
        new{T}(Int[], 0., [zeros(0,0,0,0)])
    end
end


function add_spin(ps::Partial_sol{T}, s::Int) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol{T}(vcat(ps.spins, [s]), T(0.), ps.upper_mps)
end


function solve(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T, threshold::T = T(0.)) where T <: AbstractFloat
    problem_size = maximum(grid)
    s = size(grid)
    M = make_pepsTN(grid, qubo, β)

    partial_solutions = Partial_sol{T}[Partial_sol{T}()]

    for row in 1:s[1]

        #this may need to ge cashed
        lower_mps = make_lower_mps(M, row + 1, threshold)

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


function reduce_bond_size_svd_right2left(t1::Array{T, 4}, t2::Array{T, 4}, threshold::T) where T <: AbstractFloat

    s = size(t1)

    p1 = [2,1,3,4]
    A1 = permutedims(t1, p1)
    A1 = reshape(A1, (s[2], s[1]*s[3]*s[4]))

    U,Σ,V = svd(A1)
    k = length(filter(e -> e > threshold, Σ))
    proj = transpose(U)[1:k,:]

    @tensor begin
        t1[a,e,c,d] := t1[a,b,c,d]*proj[e,b]
    end

    @tensor begin
        t2[e,a,c,d] := t2[b,a,c,d]*proj[e,b]
    end
    t1, t2
end


function reduce_bond_size_svd_left2right(t1::Array{T, 4}, t2::Array{T, 4}, threshold::T) where T <: AbstractFloat

    s = size(t2)
    A2 = reshape(t2, (s[1], s[2]*s[3]*s[4]))

    U,Σ,V = svd(A2)
    k = length(filter(e -> e > threshold, Σ))
    proj = transpose(U)[1:k,:]

    @tensor begin
        t1[a,e,c,d] := t1[a,b,c,d]*proj[e,b]
    end

    @tensor begin
        t2[e,a,c,d] := t2[b,a,c,d]*proj[e,b]
    end
    t1, t2
end


function svd_approx(mps::Vector{Array{T, 4}}, threshold::T) where T <: AbstractFloat
    for i in 1:(length(mps)-1)
        mps[i], mps[i+1] = reduce_bond_size_svd_right2left(mps[i], mps[i+1], threshold)

        #println("left 2 right")
        #println(" i = ", i, "i+1 = ", i+1)
        #print_tensors_squared(mps[i], mps[i+1])
    end

    for i in length(mps):-1:2
        mps[i-1], mps[i] = reduce_bond_size_svd_left2right(mps[i-1], mps[i], threshold)

        #println("right 2 left")
        #println(" i = ", i-1, "i+1 = ", i)
        #print_tensors_squared(mps[i-1], mps[i])
    end
    mps
end


function print_tensors_squared(t1, t2)
    A = permutedims(t1, [2,1,3,4])
    s = size(A)
    A = reshape(A, (s[1], s[2]*s[3]*s[4]))

    println("left =", A*A')

    A = t2
    s = size(A)
    A = reshape(A, (s[1], s[2]*s[3]*s[4]))

    println("right = ", A*A')
end

grid = [1 2 3; 4 5 6; 7 8 9]

function make_qubo()
    css = -2.
    qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
    qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
    qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

# it is just a concept it will be changed


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



function compute_scalar_prod1(mps_down::Vector{Array{T, 4}}, mps_up::Vector{Array{T, N} where N}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps_up):-1:1
        env = scalar_prod_step(mps_down[i], mps_up[i], env)
    end
    #size(env) == (1,1) || error("output size $(size(env)) ≠ (1,1) not fully contracted")
    env
end

if false

qubo = make_qubo()

M = make_pepsTN(grid, qubo, 1.)

M[1,2][:,:,:,:,2]

lower_mps = make_lower_mps(M, 2, 0.)

upper = [M[1,1], sum_over_last(M[1,2]), sum_over_last(M[1,3])]

upper_p = [ones(1,2,1,2), M[1,2], sum_over_last(M[1,3])]


a = compute_scalar_prod1(lower_mps, upper)[1,1,:]
a = a./sum(a)

b = compute_scalar_prod1(lower_mps, upper_p)[1,1,:]

b[1]*a[1]
b[2]*a[1]
b[1]*a[2]
b[2]*a[2]

solve(qubo, grid, 2; β = 1.)
end
