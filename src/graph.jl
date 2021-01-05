export factor_graph #decompose_edges!
export Cluster, rank_reveal

const SimpleEdge = LightGraphs.SimpleGraphs.SimpleEdge
const EdgeIter = Union{LightGraphs.SimpleGraphs.SimpleEdgeIter, Base.Iterators.Filter, Array}

mutable struct Cluster
    tag::Int
    vertices::Dict{Int, Int}
    edges::EdgeIter
    rank::Vector
    J::Matrix{<:Number}
    h::Vector{<:Number}

    function Cluster(ig::MetaGraph, v::Int)
        cl = new(v)
        active = filter_vertices(ig, :active, true)

        if cl.tag == 0
            vlist = vertices(ig)
        else
            vlist = filter_vertices(ig, :cell, v)
        end
        vlist = intersect(active, vlist)

        L = length(collect(vlist))
        cl.h = zeros(L)
        cl.J = zeros(L, L)

        cl.vertices = Dict()
        cl.edges = SimpleEdge[]

        rank = get_prop(ig, :rank)
        cl.rank = rank[1:L]

        for (i, w) ∈ enumerate(vlist)
            push!(cl.vertices, w => i)
            @inbounds cl.h[i] = get_prop(ig, w, :h)
            @inbounds cl.rank[i] = rank[w]
        end

        for e ∈ edges(ig)
            if src(e) ∈ vlist && dst(e) ∈ vlist
                i, j = cl.vertices[src(e)], cl.vertices[dst(e)]
                @inbounds cl.J[i, j] = get_prop(ig, e, :J)
                push!(cl.edges, e)
            end
        end
        cl
    end
end

function MetaGraphs.filter_edges(ig::MetaGraph, v::Cluster, w::Cluster)
    edges = SimpleEdge[]
    for i ∈ keys(v.vertices), j ∈ neighbors(ig, i)
        if j ∈ keys(w.vertices) push!(edges, SimpleEdge(i, j)) end
    end
    edges
end

mutable struct Edge
    tag::NTuple{2, Int}
    edges::EdgeIter
    J::Matrix{<:Number}

    function Edge(ig::MetaGraph, v::Cluster, w::Cluster)
        ed = new((v.tag, w.tag))
        ed.edges = filter_edges(ig, v, w)

        m = length(v.vertices)
        n = length(w.vertices)

        ed.J = zeros(m, n)
        for e ∈ ed.edges
            i = v.vertices[src(e)]
            j = w.vertices[dst(e)]
            @inbounds ed.J[i, j] = get_prop(ig, e, :J)
        end
        ed
    end
end

function _mv(ig::MetaGraph)
    L = 0
    for v ∈ vertices(ig)
        L = max(L, get_prop(ig, v, :cell))
    end
    L
end

function factor_graph(
    ig::MetaGraph;
    energy::Function=energy,
    spectrum::Function=full_spectrum,
)
    L = _mv(ig)
    fg = MetaDiGraph(L, 0.0)

    for v ∈ vertices(fg)
        cl = Cluster(ig, v)
        set_prop!(fg, v, :cluster, cl)

        sp = spectrum(cl)
        set_prop!(fg, v, :spectrum, sp)
        set_prop!(fg, v, :loc_en, vec(sp.energies))
    end

    for i ∈ 1:L, j ∈ i+1:L
        v = get_prop(fg, i, :cluster)
        w = get_prop(fg, j, :cluster)

        edg = Edge(ig, v, w)
        if !isempty(edg.edges)
            e = SimpleEdge(i, j)

            add_edge!(fg, e)
            set_prop!(fg, e, :edge, edg)

            pl, en = rank_reveal(energy(fg, edg), :PE)
            en, pr = rank_reveal(en, :EP)
            set_prop!(fg, e, :decomposition, (pl, en, pr))
        end
    end
    fg
end


# needs to be rewritten!
function rank_reveal(energy, order=:PE)
    @assert order ∈ (:PE, :EP)
    dim = order == :PE ? 1 : 2

    E = unique(energy, dims=dim)
    idx = indexin(eachslice(energy, dims=dim), collect(eachslice(E, dims=dim)))

    P = order == :PE ? zeros(size(energy, 1), size(E, 1)) : zeros(size(E, 2), size(energy, 2))

    for (i, elements) ∈ enumerate(eachslice(P, dims=dim))
        elements[idx[i]] = 1
    end

    order == :PE ? (P, E) : (E, P)
end
