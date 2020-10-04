struct PEPS
    tensors
end

struct PEPSTensor{T <: AbstractArray{<:Number, 5}}
    cluster::Cluster
    cluster_state::Vector{Int}
    data::T
end

struct PEPS
    tensors::Matrix{PEPSTensor}
end