struct PEPSTensor{T <: AbstractArray{<:Number, 5}}
    cluster::Cluster
    cluster_state::Vector{Int}
    data::T
end

struct PEPS
    tensors::Matrix{PEPSTensor}
end

struct spectrum
    enum::Vector
    states::Vector{State}
end

#=
function peps(c::Chimera, node::Int)
    for v ∈ unique_neighbors(c.graph, node)
        outer = filter_edges(c.graph, :outer, (node, v))
    end
end
=#