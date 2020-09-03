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

function make_peps_node(struct_M::Matrix{Int}, qubo::Vector{Qubo_el{T}}, i::Int, β::T) where T <: AbstractFloat

    ind = findall(x->x==i, struct_M)[1]
    h = filter(x->x.ind==(i,i), qubo)[1].coupling
    bonds = [[0], [0], [0], [0], [-1,1]]

    # determine bonds directions from struct_M
    l = 0 < ind[2]-1
    r = ind[2]+1 <= size(struct_M, 2)
    u = 0 < ind[1]-1
    d = ind[1]+1 <= size(struct_M, 1)

    if l
        bonds[1] = [-1,1]
    end

    Jir = T(0.)
    if r
        j = struct_M[ind[1], ind[2]+1]
        Jir = JfromQubo_el(qubo, i,j)
        bonds[2] = [-1,1]
    end

    if u
        bonds[3] = [-1,1]
    end

    Jid = T(0.)
    if d
        j = struct_M[ind[1]+1, ind[2]]
        Jid = JfromQubo_el(qubo, i,j)
        bonds[4] = [-1,1]
    end

    tensor_size = make_tensor_sizes(l,r,u,d,2,2)
    tensor = zeros(T, tensor_size)

    for i in CartesianIndices(tensor_size)
        b = [bonds[j][i[j]] for j in 1:5]
        tensor[i] = Tgen(b..., Jir, Jid, h, β)
    end
    tensor
end

# tensor network

function make_pepsTN(struct_M::Matrix{Int}, qubo::Vector{Qubo_el{T}}, β::T) where T <: AbstractFloat
    s = size(struct_M)
    M_of_tens = Array{Union{Nothing, Array{T}}}(nothing, s)
    for i in 1:prod(s)
        ind = findall(x->x==i, struct_M)[1]
        M_of_tens[ind] = make_peps_node(struct_M, qubo, i, β)
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
            if i > 1
                # breaks bonds between subsequent tensors in row
                # if s is set, excludes first element of the row
                output_mps[i] = A[ind:ind,:,:,:]
                output_mps[i-1] = output_mps[i-1][:,ind:ind,:,:]
            else
                output_mps[i] = A
            end
            if size(A, 3) > 1
                up_bonds[i] = ind
                output_mps[i] = output_mps[i][:,:,ind:ind,:]
            end
        end
    end
    Vector{Array{T, 4}}(output_mps), up_bonds
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
    mpo, s = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)
    reduce_bonds_horizontally!(mps_u, s)
    mps_n = MPSxMPO(mpo, mps_u)
    compute_scalar_prod(mps_d, mps_n), mps_n
end

function comp_marg_p_first(mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mps_u, _ = set_spins_on_mps(M, ses)
    compute_scalar_prod(mps_d, mps_u), mps_u
end

function comp_marg_p_last(mps_u::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo, s = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)
    reduce_bonds_horizontally!(mps_u, s)
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


function solve(qubo::Vector{Qubo_el{T}}, struct_M::Matrix{Int}, no_sols::Int = 2; β::T, threshold::T = T(0.)) where T <: AbstractFloat
    problem_size = maximum(struct_M)
    s = size(struct_M)
    M = make_pepsTN(struct_M, qubo, β)

    partial_solutions = Partial_sol{T}[Partial_sol{T}()]

    for row in 1:s[1]

        #this may need to ge cashed
        lower_mps = make_lower_mps(M, row + 1, threshold)

        p = sortperm(struct_M[row,:])
        for j in struct_M[row,p]

             a = [add_spin(ps, 1) for ps in partial_solutions]
             b = [add_spin(ps, -1) for ps in partial_solutions]
             partial_solutions = vcat(a,b)

             for ps in partial_solutions

                part_sol = ps.spins
                sol = part_sol[1+(row-1)*s[2]:end]
                l = s[2] - length(sol)
                sol = vcat(sol, fill(0, l))
                sol = sol[p]

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
