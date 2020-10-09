export Edge, Node, Grid

const Node = Int
const Edge = Tuple{Node, Node}

const IsingInstance = Dict{Edge, Float64}

struct Cluster
    instance::IsingInstance
    nodes::Vector{Node}
    legs::Dict{Symbol, Dict{Node, Vector{Node}}}
end
