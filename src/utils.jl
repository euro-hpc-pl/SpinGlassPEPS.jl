export _toIdx, _idx, _toIsing
# export newdim

# newdim(::Type{T}, dims) where {T<:AbstractArray} = T.name.wrapper{eltype(T), dims}

_toIsing(state::Vector{Int}) = 2 .* state .- 1

const _idx = Dict(-1 => 1, 1 => 2)

function _toIdx(s::Int) 
    if s == -1 
        return 1
    elseif s == 1
        return 2  
    else
        error("Wrong value $(s).")
    end          
end 
