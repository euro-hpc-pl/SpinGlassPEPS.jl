export ising_graph
export energy, rank_vec
export Cluster, Spectrum, cluster, rank

const Instance = Union{String, Dict}
const SimpleEdge = LightGraphs.SimpleGraphs.SimpleEdge
const EdgeIter = Union{LightGraphs.SimpleGraphs.SimpleEdgeIter, Base.Iterators.Filter, Array}

struct Spectrum
    energies::Array{<:Number}
    states::Array{Vector{<:Number}}
end

unique_vertices(ising_tuples) = sort(collect(Set(Iterators.flatten((i, j) for (i, j, _) ∈ ising_tuples))))

"""
$(TYPEDSIGNATURES)

Create the Ising spin glass model.

# Details

Store extra information
"""
function ising_graph(
    instance::Instance,
    sgn::Number=1.0,
    rank_override::Dict{Int, Int}=Dict{Int, Int}()
)
    # load the Ising instance
    if instance isa String
        ising = CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#")
    else
        ising = [ (i, j, J) for ((i, j), J) ∈ instance ]
    end

    original_verts = unique_vertices(ising)
    verts_map = Dict(w => i for (i, w) ∈ enumerate(original_verts))

    L = length(original_verts)

    ig = MetaGraph(L)

    foreach(args -> set_prop!(ig, args[1], :orig_vert, args[2]), enumerate(original_verts))

    J = zeros(L, L)
    h = zeros(L)

    # setup the model (J_ij, h_i)
    for (_i, _j, v) ∈ ising
        i, j = verts_map[_i], verts_map[_j]
        v *= sgn

        if i == j
            h[i] = v
        else
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || throw(ArgumentError("Duplicate Egde ($i, $j)"))
            J[i, j] = v
        end
    end

    foreach(i -> set_prop!(ig, i, :h, h[i]), vertices(ig))

    set_prop!(
        ig,
        :rank,
        Dict{Int, Int}(
            verts_map[v] => get(rank_override, v, 2) for v in original_verts
        )
    )

    set_prop!(ig, :J, J)
    set_prop!(ig, :h, h)

    ig
end

rank_vec(ig::MetaGraph) = collect(values(get_prop(ig, :rank)))

mutable struct Cluster
    vertices::Dict{Int, Int}
    edges::EdgeIter
    rank::Vector
    J::Matrix{<:Number}
    h::Vector{<:Number}
end

function cluster(ig::MetaGraph, verts::Set{Int})
    sub_ig, vmap = induced_subgraph(ig, collect(verts))

    h = get_prop.(Ref(sub_ig), vertices(sub_ig), :h)
    rank = get_prop.(Ref(sub_ig), vertices(sub_ig), :rank)
    J = get_prop(ig, :J)[vmap, vmap]

    set_prop!(sub_ig, :rank, rank)
    set_prop!(sub_ig, :J, J)
    set_prop!(sub_ig, :h, h)

    sub_ig
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
