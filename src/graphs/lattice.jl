export Lattice
export unit_cell

mutable struct Lattice <: Model
    size::NTuple{2, Int}
    graph::MetaGraph

    function Lattice(m::Int, n::Int)
        lt = new()
        lt.size = (m, n)
        lt.graph = MetaGraph(grid([m, n]))
        lt
    end
end

function Base.getindex(l::Lattice, i::Int, j::Int)
    m, n = size(l)
    LinearIndices((1:m, 1:n))[i, j]
end
