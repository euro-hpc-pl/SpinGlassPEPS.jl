using TensorOperations

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
c(γ::Int, J::T, s::Int, β::T) where T <: AbstractFloat =  exp(-β*J*γ*s)

"""
    function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jir::T, Jid::T, Jii::T , β::T) where T <: AbstractFloat

returns the element of the tensor, l, r, u, d represents the link with another tensor in the grid.
If some of these are zero there is no link and the corresponding boulding block returns 1.
"""
function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jil::T, Jiu::T, Jii::T, β::T) where T <: AbstractFloat
    delta(r, s)*delta(d, s)*c(l, Jil, s, β)*c(u, Jiu, s, β)*exp(-β*Jii*s)
end


"""
    function sum_over_last(T::Array{T, N}) where N

    used to trace over the phisical index, treated as the last index

"""
function sum_over_last(tensor::Array{T, N}) where {T <: AbstractFloat, N}
    tensorcontract(tensor, collect(1:N), ones(size(tensor,N)), [N])
end

"""
    set_last(T::Array{T, N}, s::Int) where N

set value of the physical index, s ∈ {-1,1} are supported
"""
function set_last(tensor::Array{T, N}, s::Int) where {T <: AbstractFloat, N}
    if s == -1
        B = [1.,0.]
    elseif s == 1
        B = [0.,1.]
    else
        error("spin value $s ∉ {-1, 1}")
    end
    tensorcontract(tensor, collect(1:N), B, [N])
end


function spins2index(s::Int)
    s in [-1, 1] || error("spin must equal to -1 or 1 we get $s")
    div((s+1), 2)+1
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
