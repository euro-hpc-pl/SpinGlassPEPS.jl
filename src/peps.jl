using TensorOperations

include("notation.jl")

function make_pepsTN(struct_M::Matrix{Int}, qubo::Vector{Qubo_el}, T::Type = Float64)
    s = size(struct_M)
    M_of_tens = Array{Union{Nothing, Array{T}}}(nothing, s)
    for i in 1:prod(s)
        ind = findall(x->x==i, struct_M)[1]
        M_of_tens[ind] = make_peps_node(struct_M, qubo, i, T)
    end
    Array{Array{T, 5}}(M_of_tens)
end


function JfromQubo_el(qubo::Vector{Qubo_el}, i::Int, j::Int)
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

function make_peps_node(struct_M::Matrix{Int}, qubo::Vector{Qubo_el}, i::Int, T::Type = Float64)

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

    Jir = 0.
    if r
        j = struct_M[ind[1], ind[2]+1]
        Jir = JfromQubo_el(qubo, i,j)
        bonds[2] = [-1,1]
    end

    if u
        bonds[3] = [-1,1]
    end

    Jid = 0.
    if d
        j = struct_M[ind[1]+1, ind[2]]
        Jid = JfromQubo_el(qubo, i,j)
        bonds[4] = [-1,1]
    end

    tensor_size = make_tensor_sizes(l,r,u,d,2,2)
    tensor = zeros(T, tensor_size)

    for i in CartesianIndices(tensor_size)
        b = [bonds[j][i[j]] for j in 1:5]
        tensor[i] = Tgen(b..., Jir, Jid, h)
    end
    tensor
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
    reduce_bonds_horizontally!(mps_u, s)
    mps_n = MPSxMPO(mpo, mps_u)
    compute_scalar_prod(mps_d, mps_n)
end

function comp_marg_p_first(mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo, _ = set_spins_on_mps(M, ses)
    compute_scalar_prod(mps_d, mpo)
end

function comp_marg_p_last(mps_u::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo, s = set_spins_on_mps(M, ses)
    reduce_bonds_horizontally!(mps_u, s)
    compute_scalar_prod(mpo, mps_u)
end

if false

# it has to be finished
function solve(qubo::Vector{Qubo_el}, struct_M::Matrix{Int}, no_states::Int = 2)
    problem_size = maximum(struct_M)
    M = make_pepsTN(struct_M, qubo)


    part_sol = zeros(Int, 2,1)
    part_sol[1,1] = 1
    part_sol[2,1] = -1
    for j in 1:problem_size
        ind = findall(x->x==j, struct_M)[1]
        row = ind[1]
        s = size(struct_M, 1)
        objective = Float64[]
        for i in 1:size(part_sol,1)
            r = 1
            push!(objective, r)
        end
        p = sortperm(objective)
        p1, k = get_last_m(p, no_states)
        a_temp = zeros(Int, k, j)
        for i in 1:k
            a_temp[i,:] = part_sol[p1[i],:]
        end
        if j == problem_size
            return a_temp, objective[p1]
        else
            part_sol = add_another_spin2configs(part_sol)
        end
    end
    0
end


function make_qubo()
    qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
    qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
    qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

qubo = make_qubo()

struct_M = [1 2 3; 6 5 4; 7 8 9]
s = size(struct_M)
ses = [1,-1,1]
l = length(ses)
ind = findall(x->x==l, struct_M)[1]
collect(1:ind[1]-1)
collect(ind[1]+1:s[2])





end
