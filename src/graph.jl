export Chimera, Lattice
export factor_graph, decompose_edges!
export Cluster, Spectrum
export rank_reveal



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

    @eval MetaGraphs.$op(c::Graph, args...) = $op(c.graph, args...)
end
@inline has_edge(g::Graph, x...) = has_edge(g.graph, x...)

Base.size(c::Graph) = c.size
Base.size(c::Graph, i::Int) = c.size[i]

function Base.getindex(c::Chimera, i::Int, j::Int, u::Int, k::Int)
    _, n, t = size(c)
    t * (2 * (n * (i - 1) + j - 1) + u - 1) + k
end

function Base.getindex(l::Lattice, i::Int, j::Int)
    m, n, _ = size(l)
    LinearIndices((1:m, 1:n))[i, j]
end

function Base.getindex(c::Chimera, i::Int, j::Int)
    t = size(c, 3)
    idx = vec([c[i, j, u, k] for u=1:2, k=1:t])
    c.graph[idx]
end

function unit_cell(l::Lattice, v::Int)
    Cluster(l.graph, v, enum([v]), [])
end

function unit_cell(c::Chimera, v::Int)
    elist = filter_edges(c.graph, :cells, (v, v))
    vlist = filter_vertices(c.graph, :cell, v)
    Cluster(c.graph, v, enum(vlist), elist)
end

Cluster(g::Graph, v::Int) = unit_cell(g, v)

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

function factor_graph(
    g::Graph, 
    energy::Function=ising_energy, 
    spectrum::Function=brute_force, 
    create_cluster::Function=Cluster,
) # how to add typing to functions?
    m, n, _ = g.size
    fg = factor_graph(m, n)

    for v ∈ vertices(fg)
        cl = create_cluster(g, v)
        set_prop!(fg, v, :cluster, cl)
        set_prop!(fg, v, :spectrum, spectrum(cl))
    end

    for e ∈ edges(fg)
        v = get_prop(fg, src(e), :cluster)
        w = get_prop(fg, dst(e), :cluster)

        edge = Edge(g.graph, v, w)
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
            dec = (p, exp.(beta .* en))
        else
            dec = (exp.(beta .* en), p)
        end
        set_prop!(fg, edge, :decomposition, dec)
    end 
end
 

function rank_reveal(energy, order=:PE) # or E_then_P
    dim = order == :PE ? 1 : 2

    E = unique(energy, dims=dim)
    idx = indexin(eachslice(energy, dims=dim), collect(eachslice(E, dims=dim)))

    P = order == :PE ? zeros(size(energy, 1), size(E, 1)) : zeros(size(E, 2), size(energy, 2))
    
    for (i, elements) ∈ enumerate(eachslice(P, dims=dim))
        elements[idx[i]] = 1
    end

    order == :PE ? (P, E) : (E, P)
end 