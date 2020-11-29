export Chimera
struct Chimera
    size::NTuple{3, Int}
    graph::MetaGraph
end

function Chimera(m::Int, n::Int=m, t::Int=4)
    max_size = m * n * 2 * t
    g = MetaGraph(max_size)

    hoff = 2t
    voff = n * hoff
    mi = m * voff
    ni = n * hoff

    for i=1:hoff:ni, j=i:voff:mi, k0=j:j+t-1, k1=j+t:j+2t-1
        add_edge!(g, k0, k1)
    end

    for i=t:2t-1, j=i:hoff:ni-hoff-1, k=j+1:voff:mi-1
        add_edge!(g, k, k+hoff-1)
    end

    for i=1:t, j=i:hoff:ni-1, k=j:voff:mi-voff-1
        add_edge!(g, k, k+voff)
    end
    Chimera((m, n, t), g)
end

for op in [
    :nv,
    :ne,
    :eltype,
    :edgetype,
    :vertices,
    :edges,
    ]

    @eval LightGraphs.$op(c::Chimera) = $op(c.graph)
end

for op in [
    :get_prop,
    :set_prop!,
    :has_vertex,
    :inneighbors,
    :outneighbors,
    :neighbors]

    @eval MetaGraphs.$op(c::Chimera, args...) = $op(c.graph, args...)
end
@inline has_edge(g::Chimera, x...) = has_edge(g.graph, x...)

Base.size(c::Chimera) = c.size
Base.size(c::Chimera, i::Int) = c.size[i]

function Base.getindex(c::Chimera, i::Int, j::Int, u::Int, k::Int)
    _, n, t = size(c)
    t * (2 * (n * (i - 1) + j - 1) + u - 1) + k
end

function Base.getindex(c::Chimera, i::Int, j::Int)
    t = size(c, 3)
    idx = vec([c[i, j, u, k] for u=1:2, k=1:t])
    c.graph[idx]
end

# """
# $(TYPEDSIGNATURES)

# Returns Chimera's unit cell at \$(i, j)\$ on a grid.

# """
# function chimera_cell(chimera::Chimera, i::Int, j::Int)
#     N, M, T = chimera.size
#     C = N * M

#     @assert i <= N & j <= M "Cell ($i, $j) is outside the graph C$C."

#     vlist = [ chimera_to_linear(i, j, u, k) for u ∈ 1:T for k ∈ [0, 1] ]
#     cell, vmap = induced_subgraph(chimera.graph, vlist)

#     for (i, v) ∈ zip(nv(cell), vmap)
#         set_prop!(cell, i, :global, v)

#         nbrs = unique_neighbors(chimera.graph, v)

#         for w ∈ intersect(vlist, nbrs)
#             J = get_prop(chimera.graph, v, w, :J)
#             set_prop!(cell, i, j, :J, J)
#         end

#         h = get_prop(chimera.graph, v, :h)
#         set_prop!(cell, i, :h, h)
#     end
    
#     set_prop!(cell, :description, "Unit cell ($i, $j) of Chimera $C.") 
#     cell
# end
