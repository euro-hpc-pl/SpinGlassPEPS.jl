mutable struct Lattice <: Model
    size::NTuple{3, Int}
    graph::MetaGraph

    function Lattice(m::Int, n::Int, k::Int=0)
        lt = new()
        lt.size = (m, n, k)
        lt.graph = MetaGraph(grid([m, n]))
        lt
    end
end