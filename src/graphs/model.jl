export Model

abstract type Model end

for op in [
    :nv,
    :ne,
    :eltype,
    :edgetype,
    :vertices,
    :edges,
    ]

    @eval LightGraphs.$op(c::Model) = $op(c.graph)
end

for op in [
    :get_prop,
    :set_prop!,
    :has_vertex,
    :inneighbors,
    :outneighbors,
    :neighbors]

    @eval MetaGraphs.$op(c::Model, args...) = $op(c.graph, args...)
end
@inline has_edge(g::Model, x...) = has_edge(g.graph, x...)

Base.size(c::Model) = c.size
Base.size(c::Model, i::Int) = c.size[i]

Cluster(g::Model, v::Int) = unit_cell(g, v)