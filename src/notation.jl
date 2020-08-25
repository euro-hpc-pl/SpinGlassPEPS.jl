# we set β as a global variable, at lest now
β = 1.

struct Qubo_el
    ind::Tuple{Int, Int}
    coupling::Float64
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
    c(γ::Int, J::Float64, s::Int)

c building block
"""
c(γ::Int, J::Float64, s::Int) =  exp(-β*J*γ*s)

"""
    function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jir::Float64, Jid::Float64, Jii::Float64)

returns the element of the tensor, l, r, u, d represents the link with another tensor in the grid.
If some of these are zero there is no link and the corresponding boulding block returns 1.
"""
function Tgen(l::Int, r::Int, u::Int, d::Int, s::Int, Jir::Float64, Jid::Float64, Jii::Float64)
    delta(l, s)*delta(u, s)*c(r, Jir, s)*c(d, Jid, s)*exp(-β*Jii*s)
end



"""
    function sum_over_last(T::Array{Float64, N}) where N

    used to trace over the phisical index, treated as the last index

"""
function sum_over_last(T::Array{Float64, N}) where N
    tensorcontract(T, collect(1:N), ones(size(T,N)), [N])
end

"""
    set_last(T::Array{Float64, N}, s::Int) where N

set value of the physical index, s ∈ {-1,1} are supported
"""
function set_last(T::Array{Float64, N}, s::Int) where N
    if s == -1
        B = [1.,0.]
    elseif s == 1
        B = [0.,1.]
    else
        error("spin value $s ∉ {-1, 1}")
    end
    tensorcontract(T, collect(1:N), B, [N])
end


function spins2index(s::Int)
    s in [-1, 1] || error("spin must equal to -1 or 1 we get $s")
    div((s+1), 2)+1
end



"""
    get_last_m(vector::Vector{Int}, m::Int)

returns last m element of the vector{Int} or the whole vector if it has less than m elements
returns as well the size of the returned vector
"""
function get_last_m(vector::Vector{Int}, m::Int)
    if length(vector) <= m
        return vector, length(vector)
    else
        return vector[end-m+1:end], m
    end
end



"""
    add_another_spin2configs(configs::Matrix{Int})

given the size(configs,1) configurations of size(configs,2) spins
add another spin to the end in all configurations.

Return matrix of size  2*size(configs,1), size(configs,2)+1
"""
function add_another_spin2configs(configs::Matrix{Int})
    s = size(configs)
    ret = vcat(configs, configs)
    ses = vcat(fill(-1, s[1]), fill(1, s[1]))
    hcat(ret, ses)
end
