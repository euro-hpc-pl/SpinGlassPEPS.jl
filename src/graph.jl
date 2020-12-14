export Chimera, factor_graph
export Cluster, Spectrum, PepsTensor

@enum TensorsOrder begin
    P_then_E = 1
    E_then_P = -1
end

@enum HorizontalDirections begin
    left_to_right = 1
    right_to_left = -1
end

@enum VerticalDirections begin
    top_to_bottom = -1
    bottom_to_top = 1
end

mutable struct Chimera
    size::NTuple{3, Int}
    graph::MetaGraph
    
    function Chimera(size::NTuple{3, Int}, graph::MetaGraph)
        cg = new(size, graph)
        m, n, t = size
        linear = LinearIndices((1:m, 1:n))

        for i=1:m, j=1:n, u=1:2, k=1:t
            v = cg[i, j, u, k]
            ij = linear[i, j]
            set_prop!(cg, v, :cell, ij)
        end

        for e in edges(cg)
            v = get_prop(cg, src(e), :cell)
            w = get_prop(cg, dst(e), :cell)
            set_prop!(cg, e, :cells, (v, w))
        end
        cg
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

function unit_cell(c::Chimera, v::Int)
    elist = filter_edges(c.graph, :cells, (v, v))
    vlist = filter_vertices(c.graph, :cell, v)
    Cluster(c.graph, v, enum(vlist), elist)
end

Cluster(c::Chimera, v::Int) = unit_cell(c, v)

#Spectrum(cl::Cluster) = brute_force(cl, num_states=256)

function Spectrum(cl::Cluster)
    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    Spectrum(energies, σ)   
end

function factor_graph(m::Int, n::Int, hdir=left_to_right, vdir=bottom_to_top)
    dg = MetaGraph(SimpleDiGraph(m * n))
    set_prop!(dg, :order, (hdir, vdir))

    linear = LinearIndices((1:m, 1:n))
    for i ∈ 1:m
        for j ∈ 1:n-1
            v, w = linear[i, j], linear[i, j+1]
            Int(hdir) == 1 ? e = SimpleEdge(v, w) : e = SimpleEdge(w, v)
            add_edge!(dg, e)
            set_prop!(dg, e, :orientation, "horizontal")
        end
    end

    for i ∈ 1:n
        for j ∈ 1:m-1
            v, w = linear[i, j], linear[i, j+1]
            Int(vdir) == 1 ? e = SimpleEdge(v, w) : e = SimpleEdge(w, v)
            add_edge!(dg, e)
            set_prop!(dg, e, :orientation, "vertical")
        end
    end
    dg
end

function factor_graph(c::Chimera)
    m, n, _ = c.size
    fg = factor_graph(m, n)

    for v ∈ vertices(fg)
        cl = Cluster(c, v)
        set_prop!(fg, v, :cluster, cl)
        set_prop!(fg, v, :spectrum, Spectrum(cl))
    end

    for e ∈ edges(fg)
        v = get_prop(fg, src(e), :cluster)
        w = get_prop(fg, dst(e), :cluster)

        edge = Edge(c.graph, v, w)
        set_prop!(fg, e, :edge, edge)
        set_prop!(fg, e, :energy, energy(fg, edge))
    end
    fg
end


function decompose_edges!(fg::MetaGraph, order=P_then_E, beta::Float64=1.0)
    set_prop!(dg, :tensorsOrder, order)

    for edge ∈ edges(fg)
        energies = get_prop(fg, edge, :energy)
        en, p = rank_reveal(energies)

        if Int(order) == 1
            dec = (exp.(beta .* en), p)
        else
            dec = (p, exp.(beta .* en))
        end
        set_prop!(fg, edge, :decomposition, dec)
    end 
end
 
