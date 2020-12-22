export Lattice
export unit_cell

struct Lattice <: Model
    size::NTuple{2, Int}
    graph::MetaGraph
end

@inline function Base.getindex(l::Lattice, i::Int, j::Int)
    m, n = size(l)
    LinearIndices((1:m, 1:n))[i, j]
end

function unit_cell(c::Lattice, v::Int)
    Cluster(c.graph, v, enum([v]), [])
end
