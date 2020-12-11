export Chimera, factor_graph
export Cluster, Spectrum

@enum HorizontalDeirections begin
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

function factor_graph(
    m::Int, 
    n::Int, 
    hdir::HorizontalDeirections=HorizontalDeirections.left_to_right, 
    vdir::VerticalDirections=VerticalDirections.bottom_to_top)
    dg = MetaGraph(SimpleDiGraph(m * n))
    set_prop!(dg, :order, (hdir, vdir))

    linear = LinearIndices((1:m, 1:n))
    for i ∈ 1:m
        for j ∈ 1:n-1
            v, w = linear[i, j], linear[i, j+1]
            hdir == 1 ? e = (v, w) : e = (w, v)
            add_edge!(dg, e)
            set_prop!(dg, e, :orientation, "horizontal")
        end
    end

    for i ∈ 1:n
        for j ∈ 1:m-1
            v, w = linear[i, j], linear[i, j+1]
            vdir == 1 ? e = (v, w) : e = (w, v)
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


function decompose_edges!(fg::MetaGraph, beta::AbstractFloat)
    for edge ∈ edges(fg)
        energies = get_prop(fg, edge, :energy)
        truncated_energies, projector = rank_reveal(energies)
        exponents = exp.(beta .* btruncated_energies)
        set_prop!(fg, edge, :projector, projector)
        set_prop!(fg, edge, :exponents, exponents)
    end 
end


function peps_tensor(fg::MetaGraph, v::Int)
    T = Dict{String, AbstractArray}()
    outgoing = outneighbors(fg, v)
    incoming = inneighbours(fg, v)

    hor_outgoing = [u for u in outgoing if get_prop!(fg, (v, u), :orientation) == "horizontal"]
    hor_incoming = [u for u in incoming if get_prop!(fg, (u, v), :orientation) == "horizontal"]
    ver_outgoing = [u for u in outgoing if get_prop!(fg, (v, u), :orientation) == "vertical"]
    ver_incoming = [u for u in incoming if get_prop!(fg, (u, v), :orientation) == "vertical"]

    for w ∈ unique_neighbors(fg, v)

        #to_exp = unique(en)    

        #set_prop!(fg, e, :energy, to_exp)
        #set_prop!(fg, e, :projector, indexin(to_exp, en))

    end

    @cast A[l, r, u, d, σ] |= T["l"][l, σ] * T["r"][r, σ] * T["d"][d, σ] * T["u"][u, σ]
end
