export idx, to_ising

to_ising(state) = 2 .* state .- 1

const Pu = [[1., 0.] [0., 0.]]
const Pd = [[0., 0.] [0., 1.]]
const proj = Dict(-1 => Pd, 1 => Pu)
const _idx = Dict(-1 => 1, 1 => 2)

function _projector(state::Vector{Int})
    P = Vector{AbstractMatrix}(undef, length(state))
    for (i, σ) ∈ enumerate(state) P[i] = proj[σ] end 
    P
end 

toIsing(state::Vector{Int}) = 2 .* state .- 1

idx(s::Int) = idx(Val(s))
idx(::Val{-1}) = 1
idx(::Val{1}) = 2
