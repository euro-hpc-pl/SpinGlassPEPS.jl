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

    @eval MetaGraphs.$op(m::Model, args...) = $op(m.graph, args...)
end

@inline has_edge(m::Model, x...) = has_edge(m.graph, x...)

Base.size(m::Model) = m.size
Base.size(m::Model, i::Int) = m.size[i]