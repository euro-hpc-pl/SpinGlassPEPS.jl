export _toIdx, _idx, _toIsing, _projector
# export newdim

# newdim(::Type{T}, dims) where {T<:AbstractArray} = T.name.wrapper{eltype(T), dims}

const Pu = [[1., 0.] [0., 0.]]
const Pd = [[0., 0.] [0., 1.]]
const proj = Dict(-1 => Pd, 1 => Pu)
const _idx = Dict(-1 => 1, 1 => 2)

function _projector(state::Vector{Int})
    P = Vector{AbstractMatrix}(undef, length(state))
    for (i, σ) ∈ enumerate(state) P[i] = proj[σ] end 
    P
end 

_toIsing(state::Vector{Int}) = 2 .* state .- 1
_toIdx(σ::Int) = _idx[σ]

