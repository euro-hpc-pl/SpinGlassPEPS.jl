using TensorOperations

function get_connection_if_exists(i::Int, j::Int, k::Int, grid::Matrix{Int})
    try
        return [[i, grid[j, k]]]
    catch
        return Vector{Int}[]
    end
end



function enviroment(i::Int,j::Int,k::Int,grid)
    left = get_connection_if_exists(i, j, k-1, grid)
    right = get_connection_if_exists(i, j, k+1, grid)
    up = get_connection_if_exists(i, j-1, k, grid)
    down = get_connection_if_exists(i, j+1, k, grid)
    return left, right, up, down
end

function read_connecting_pairs(grid::Matrix{Int}, i::Int)
    a = findall(x->x==i, grid)[1]
    enviroment(i,a[1],a[2], grid)
end

struct Bond_with_other_node
    node::Int
    spins1::Vector{Int}
    spins2::Vector{Int}
end

struct Qubo_el{T<:AbstractFloat}
    ind::Tuple{Int, Int}
    coupling::T
    function(::Type{Qubo_el{T}})(ind::Tuple{Int, Int}, coupling::T1) where {T <: AbstractFloat, T1 <: AbstractFloat}
        new{T}(ind, T(coupling))
    end
    function(::Type{Qubo_el})(ind::Tuple{Int, Int}, coupling::Float64)
        new{Float64}(ind, coupling)
    end
end

function matrix2qubo_vec(M::Matrix{T}) where T <:AbstractFloat
    s = size(M)
    q_vec = Qubo_el{T}[]
    for i in 1:s[1]
        for j in 1:i
            push!(q_vec, Qubo_el((i,j), M[i,j]))
        end
    end
    q_vec
end

"""
    struct Node_of_grid

this structure is supposed to include all information about nodes on the grid.
necessary to crearte the corresponding tensor (e.g. the element of the peps).
"""
struct Node_of_grid
    i::Int
    spin_inds::Vector{Int}
    intra_struct::Vector{Vector{Int}}
    left::Vector{Vector{Int}}
    right::Vector{Vector{Int}}
    up::Vector{Vector{Int}}
    down::Vector{Vector{Int}}
    all_connections::Vector{Vector{Int}}
    bonds::Vector{Bond_with_other_node}
    #construction from the grid
    function(::Type{Node_of_grid})(i::Int, grid::Matrix{Int})
        s = size(grid)
        intra_struct = Vector{Int}[]

        left, right, up, down = read_connecting_pairs(grid, i)
        all_connections = [left..., right..., up..., down...]
        new(i, [i], intra_struct, left, right, up, down, all_connections)
    end
    #construction from matrix of matrices (a grid of many nodes)
    function(::Type{Node_of_grid})(i::Int, Mat::Matrix{Int},
                                grid::Array{Array{Int64,N} where N,2};
                                chimera::Bool = false)

        left = Vector{Int}[]
        right = Vector{Int}[]
        up = Vector{Int}[]
        down = Vector{Int}[]

        a = findall(x->x==i, Mat)[1]
        j = a[1]
        k = a[2]
        M = grid[j,k]
        # this transpose makes better ordering
        spin_inds = vec(transpose(M))
        intra_struct = Vector[]
        s = size(M)
        if ! chimera
            for i in 1:s[1]
                if length(M[i,:]) > 1
                    push!(intra_struct, M[i,:])
                end
            end
            for i in 1:s[2]
                if length(M[:,i]) > 1
                    push!(intra_struct, M[:,i])
                end
            end
        else
            for i in 1:s[1]
                for j in 1:s[1]
                    push!(intra_struct, [M[i, 1], M[j,end]])
                end
            end

        end
        l,r,u,d = read_connecting_pairs(Mat, i)

        if l != Array{Int64,1}[]

            Mp = grid[j, k-1]
            v2 = Mp[:,end]
            v1 = 0

            if chimera
                v1 = M[:,end]
            else
                v1 = M[:,1]
            end

            for i in 1:length(v1)
                push!(left, [v1[i], v2[i]])
            end
        end
        if r != Array{Int64,1}[]
            v1 = M[:,end]
            v2 = 0

            Mp = grid[j, k+1]

            if chimera
                v2 = Mp[:,end]
            else
                v2 = Mp[:,1]
            end

            for i in 1:length(v1)
                push!(right, [v1[i], v2[i]])
            end
        end
        if u != Array{Int64,1}[]
            v1 = 0
            v2 = 0
            Mp = grid[j-1, k]
            if chimera
                v1 = M[:,1]
                v2 = Mp[:,1]
            else
                v1 = M[1,:]
                v2 = Mp[end,:]
            end
            for i in 1:length(v1)
                push!(up, [v1[i], v2[i]])
            end
        end
        if d != Array{Int64,1}[]
            v1 = 0
            v2 =0

            Mp = grid[j+1, k]
            if chimera
                v1 = M[:,1]
                v2 = Mp[:,1]
            else
                v1 = M[end,:]
                v2 = Mp[1,:]
            end
            for i in 1:length(v1)
                push!(down, [v1[i], v2[i]])
            end
        end
        #TODO this all_connections may need to be changed
        all_connections = [left..., right..., up..., down...]
        new(i, spin_inds, intra_struct, left, right, up, down, all_connections)
    end
    # construction from qubo directly, It will not check if qubo fits grid
    function(::Type{Node_of_grid})(i::Int, qubos::Vector{Qubo_el{T}}) where T <: AbstractFloat
        x = Vector{Int}[]
        all_connections = Vector{Int}[]
        for q in qubos
            if (q.ind[1] == i && q.ind[2] != i)
                push!(all_connections, [q.ind...])
            end
            if (q.ind[2] == i && q.ind[1] != i)
                push!(all_connections, [q.ind[2], q.ind[1]])
            end
        end
        new(i, [i], x, x, x, x, x, all_connections)
    end
end

function get_system_size(qubo::Vector{Qubo_el{T}}) where T <: AbstractFloat
    size = 0
    for q in qubo
        size = maximum([size, q.ind[1], q.ind[2]])
    end
    size
end

function get_system_size(ns::Vector{Node_of_grid})
    mapreduce(x -> length(x.spin_inds), +, ns)
end


# generation of tensors

"""
    delta(a::Int, b::Int)

Dirac delta, additionally return 1 if first arguments is zero for
implementation cause
"""
function delta(γ::Int, s::Int)
    if γ != 0
        return Int(γ == s)
    end
    1
end

"""
    c(γ::Int, J::T, s::Int, β::T) where T <: AbstractFloat

c building block
"""
c(γ::Int, J::T, s::Int, β::T) where T <: AbstractFloat =  exp(β*2*J*γ*s)

"""
    function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jir::T, Jid::T, Jii::T , β::T) where T <: AbstractFloat

returns the element of the tensor, l, r, u, d represents the link with another tensor in the grid.
If some of these are zero there is no link and the corresponding boulding block returns 1.
"""
function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jil::T, Jiu::T, Jii::T, β::T) where T <: AbstractFloat
    delta(r, s)*delta(d, s)*c(l, Jil, s, β)*c(u, Jiu, s, β)*exp(β*Jii*s)
end


"""
    function sum_over_last(T::Array{T, N}) where N

    used to trace over the phisical index, treated as the last index

"""
function sum_over_last(tensor::Array{T, N}) where {T <: AbstractFloat, N}
    dropdims(sum(tensor, dims = N), dims = N)
end


"""
    last_m_els(vector::Vector{Int}, m::Int)

returns last m element of the vector{Int} or the whole vector if it has less than m elements

"""
function last_m_els(vector::Vector{Int}, m::Int)
    if length(vector) <= m
        return vector
    else
        return vector[end-m+1:end]
    end
end

"""
    JfromQubo_el(qubo::Vector{Qubo_el{T}}, i::Int, j::Int) where T <: AbstractFloat

reades the coupling from the qubo, returns the number
"""
function JfromQubo_el(qubo::Vector{Qubo_el{T}}, i::Int, j::Int) where T <: AbstractFloat
    try
        return filter(x->x.ind==(i,j), qubo)[1].coupling
    catch
        return filter(x->x.ind==(j,i), qubo)[1].coupling
    end
end

"""
    ind2spin(i::Int, s::Int = 2)

return a spin from the physical index, if size is 1, returns zero.
"""
function ind2spin(i::Int, size::Int = 2)
    if size == 1
        return [0]
    else
        s = [2^i for i in 1:ceil(Int, log(2, size))]

        return [1-2*Int((i-1)%j < j/2) for j in s]
    end
end

spins2ind(s::Int) = spins2ind([s])


function spins2ind(s::Vector{Int})
    s = [Int(el == 1) for el in s]
    v = [2^i for i in 0:1:length(s)-1]
    transpose(s)*v+1
end


function reindex(i::Int, all_spins::Vector{Int}, subset_spins::Vector{Int})
    s = ind2spin(i, 2^length(all_spins))
    k = [findall(x->x==j, all_spins)[1] for j in subset_spins]
    spins2ind(s[k])
end
