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

function trace_spins(M::Matrix{Array{T, 5}}, struct_M::Matrix{Int}, ses::Vector{Int}) where T <: AbstractFloat
     s = size(struct_M)
    M_tr = Array{Union{Nothing, Array{T}}}(nothing, s)
    for ind in CartesianIndices(s)
        k = struct_M[ind]
        if k > length(ses)
            M_tr[ind] = sum_over_last(M[ind])
        else
            M_tr[ind] = set_last(M[ind], ses[k])
        end
    end
    M_tr
end


function MPSxMPO(mps::Vector{Array{T, 4}}, mpo::Vector{Array{T, 4}}) where T <: AbstractFloat
        mps_res = Array{Union{Nothing, Array{T}}}(nothing, length(mps))
        for i in 1:length(mps)
        A = mps[i]
        B = mpo[i]
        @tensor begin
            C[a,d,b,e,f,c] := A[a,b,x,c]*B[d,e,f,x]
        end
        sa = size(A)
        sb = size(B)
        C = reshape(C, (sa[1]*sb[1], sa[2]*sb[2], sb[3], sa[4]))
        mps_res[i] = C
    end
    Array{Array{T, 4}}(mps_res)
end

function make_qubo()
    qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
    qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
    qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

if false

    qubo = make_qubo()

    struct_M = [1 2 3; 6 5 4; 7 8 9]

    M = make_pepsTN(struct_M, qubo)

    MM = trace_spins(M, struct_M, Int[])
end
