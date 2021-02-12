export ising_graph, update_cells!
export energy, rank_vec
export Cluster, Spectrum

const Instance = Union{String, Dict}
const SimpleEdge = LightGraphs.SimpleGraphs.SimpleEdge
const EdgeIter = Union{LightGraphs.SimpleGraphs.SimpleEdgeIter, Base.Iterators.Filter, Array}

struct Spectrum
    energies::Array{<:Number}
    states::Array{Vector{<:Number}}
end

function rank_vec(ig::MetaGraph)
    rank = get_prop(ig, :rank)
    L = get_prop(ig, :L)
    Int[get(rank, i, 1) for i=1:L]
end

"""
$(TYPEDSIGNATURES)

Create the Ising spin glass model.

# Details

Store extra information
"""
function ising_graph(
    instance::Instance,
    L::Int,
    sgn::Number=1.0,
    rank_override::Dict{Int, Int}=Dict{Int, Int}()
)

    # load the Ising instance
    if typeof(instance) == String
        ising = CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#")
    else
        ising = [ (i, j, J) for ((i, j), J) ∈ instance ]
    end

    ig = MetaGraph(L, 0.0)
    set_prop!(ig, :description, "The Ising model.")
    set_prop!(ig, :L, L)

    for v ∈ 1:L
        set_prop!(ig, v, :active, false)
        set_prop!(ig, v, :cell, v)
        set_prop!(ig, v, :h, 0.)
    end

    J = zeros(L, L)
    h = zeros(L)

    #r
    # setup the model (J_ij, h_i)
    for (i, j, v) ∈ ising
        v *= sgn

        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
            h[i] = v
        else
            if has_edge(ig, j, i)
                error("Cannot add ($i, $j) as ($j, $i) already exists!")
            end
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
            J[i, j] = v
        end

        set_prop!(ig, i, :active, true) || error("Cannot activate node $(i)!")
        set_prop!(ig, j, :active, true) || error("Cannot activate node $(j)!")
    end

    # store extra information
    rank = Dict{Int, Int}()
    for v in vertices(ig)
        if get_prop(ig, v, :active)
            rank[v] = get(rank_override, v, 2) # TODO: this can not be 2
        end
    end   
    set_prop!(ig, :rank, rank)

    set_prop!(ig, :J, J)
    set_prop!(ig, :h, h)

    σ = 2.0 * (rand(L) .< 0.5) .- 1.0

    set_prop!(ig, :state, σ)
    set_prop!(ig, :energy, energy(σ, ig))
    ig
end

function update_cells!(ig::MetaGraph; rule::Dict)
    for v ∈ vertices(ig)
        w = get_prop(ig, v, :cell)
        set_prop!(ig, v, :cell, rule[w])
    end
end

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

        L = length(vlist)
        cl.h = zeros(L)
        cl.J = zeros(L, L)

        cl.vertices = Dict()
        cl.edges = SimpleEdge[]

        rank = get_prop(ig, :rank)
        cl.rank = zeros(Int, L)

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

"""
$(TYPEDSIGNATURES)

Calculate the Ising energy
```math
E = -\\sum_<i,j> s_i J_{ij} * s_j - \\sum_j h_i s_j.
```
"""

energy(σ::Vector, J::Matrix, η::Vector=σ) = dot(σ, J, η)
energy(σ::Vector, h::Vector) = dot(h, σ)
energy(σ::Vector, cl::Cluster, η::Vector=σ) = energy(σ, cl.J, η) + energy(cl.h, σ)
energy(σ::Vector, ig::MetaGraph) = energy(σ, get_prop(ig, :J)) + energy(σ, get_prop(ig, :h))

function energy(fg::MetaDiGraph, edge::Edge)
    v, w = edge.tag
    vSp = get_prop(fg, v, :spectrum).states
    wSp = get_prop(fg, w, :spectrum).states

    m = prod(size(vSp))
    n = prod(size(wSp))

    en = zeros(m, n)
    for (j, η) ∈ enumerate(vec(wSp))
        en[:, j] = energy.(vec(vSp), Ref(edge.J), Ref(η))
    end
    en
end
