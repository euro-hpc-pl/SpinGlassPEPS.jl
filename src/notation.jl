using TensorOperations

function get_connection_if_exists(i::Int, j::Int, k::Int, grid::Matrix{Int})
    try
        return [[i, grid[j, k]]]
    catch
        return Vector{Int}[]
    end
end

function read_connecting_pairs(grid::Matrix{Int}, i::Int)
    a = findall(x->x==i, grid)[1]
    j = a[1]
    k = a[2]
    left = get_connection_if_exists(i, j, k-1, grid)
    right = get_connection_if_exists(i, j, k+1, grid)
    up = get_connection_if_exists(i, j-1, k, grid)
    down = get_connection_if_exists(i, j+1, k, grid)
    return left, right, up, down
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
    function(::Type{Node_of_grid})(i::Int, grid::Matrix{Int})
        s = size(grid)
        intra_struct = Vector{Int}[]

        left, right, up, down = read_connecting_pairs(grid, i)
        new(i, [i], intra_struct, left, right, up, down)
    end
end

function get_system_size(ns::Vector{Node_of_grid})
    mapreduce(x -> length(x.spin_inds), +, ns)
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
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
