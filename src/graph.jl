export Chimera, outer_connections, factor_graph
struct Cluster
    vertices
    edges::EdgeIter
end

mutable struct Chimera
    size::NTuple{3, Int}
    graph::MetaGraph
    outer_connections::Dict{Tuple, Vector}
    
    function Chimera(size::NTuple{3, Int}, graph::MetaGraph)
        c = new(size, graph)
        m, n, t = size
        linear = LinearIndices((1:m, 1:n))

        for i=1:m, j=1:n, u=1:2, k=1:t
            v = c[i, j, u, k]
            ij = linear[i, j]
            set_prop!(c, v, :cluster, ij)
        end

        outer_connections = Dict{Tuple, Vector}()
        for e in edges(c)
            src_cluster = get_prop(c, src(e), :cluster)
            dst_cluster = get_prop(c, dst(e), :cluster)
            # if src_cluster == dst_cluster
            #     set_prop!(c, e, :outer, false)
            # else
            key = (src_cluster, dst_cluster)
            set_prop!(c, e, :outer, key)
            if haskey(outer_connections, key)
                push!(outer_connections[key], e)
            else
                outer_connections[key] = [e]
            end
            # end
        end
        c.outer_connections = outer_connections
        c
    end
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


outer_connections(c::Chimera, i, j, k, l) = outer_connections(c::Chimera, (i, j), (k, l))

function outer_connections(c::Chimera, src, dst) 
    ret = get(c.outer_connections, (src, dst), [])
    if length(ret) == 0
        ret = get(c.outer_connections, (dst, src), [])
    end
    ret
end

function energy(σ::State, ig::MetaGraph, cl::Cluster; sgn::Float64=-1.0)
    for e ∈ cl.edges
        println(typeof(e), "->", e)
    end
    e = energy(σ, ig, cl.edges, sgn=sgn) 
    e += energy(σ, ig, cl.vertices, sgn=sgn)
end

function factor_graph(c::Chimera)
    m, n, t = c.size

    rank = get_prop(c.graph, :rank)
    fg = MetaGraph(grid([m, n]))

    for v ∈ vertices(fg)
        vv = filter_vertices(c.graph, :cluster, v)
        ve = filter_edges(c.graph, :outer, (v, v))

        cl = Cluster(vv, ve)
        sp = all_states(rank[collect(vv)])

        set_prop!(fg, v, :states, sp)
        set_prop!(fg, v, :cluster, cl)

        en = energy.(sp, Ref(c.graph), Ref(cl))
        set_prop!(fg, v, :energy, en)
    end

    for v ∈ vertices(fg)
        for w ∈ unique_neighbors(fg, v)
            vw = filter_edges(c.graph, :outer, (v, w))

            en = []
            for η ∈ get_prop(fg, v, :spec)
                σ = get_prop(fg, w, :spec)
                push!(en, energy.(σ, c.graph, vw, η))
            end
            set_prop!(fg, v, w, :energy, en)
        end
    end
    fg
end


