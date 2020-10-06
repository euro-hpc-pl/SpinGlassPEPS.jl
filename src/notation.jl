using TensorOperations


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
        a = findall(x->x==i, grid)[1]
        k = a[1]
        j = a[2]


        left = Vector{Int}[]
        if j > 1
            left = [[i, grid[k, j-1]]]
        end

        right = Vector{Int}[]
        if j < s[2]
            right = [[i, grid[k, j+1]]]
        end

        up = Vector{Int}[]
        if k > 1
            up = [[i, grid[k-1, j]]]
        end

        down = Vector{Int}[]
        if k < s[1]
            down = [[i, grid[k+1, j]]]
        end
        new(i, [i], intra_struct, left, right, up, down)
    end
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


function JfromQubo_el(qubo::Vector{Qubo_el{T}}, i::Int, j::Int) where T <: AbstractFloat
    try
        return filter(x->x.ind==(i,j), qubo)[1].coupling
    catch
        return filter(x->x.ind==(j,i), qubo)[1].coupling
    end
end

function ind2spin(i::Int, s::Int = 2)
    if s == 1
        return 0
    elseif s == 2
        return 2*i-3
    end
    -100
    # TODO mapping
end

spins2ind(s::Int) = div(s+3, 2)
