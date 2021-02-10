export Lattice
export unit_cell

struct Lattice <: Model
    size::NTuple{2, Int}
    graph::MetaGraph

    function Lattice(m::Int, n::Int)
        lt = new((m, n))
        lt.graph = MetaGraph(grid([m, n]))
        lt
    end
end

@inline function Base.getindex(l::Lattice, i::Int, j::Int)
    m, n = size(l)
    LinearIndices((1:m, 1:n))[i, j]
end
