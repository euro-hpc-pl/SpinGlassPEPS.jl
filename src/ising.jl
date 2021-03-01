export ising_graph
export energy, rank_vec
export Spectrum, cluster, rank

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

function cluster(ig::MetaGraph, verts::Set{Int})
    sub_ig, vmap = induced_subgraph(ig, collect(verts))

    h = get_prop.(Ref(sub_ig), vertices(sub_ig), :h)
    rank = getindex.(Ref(get_prop(ig, :rank)), vmap)
    J = get_prop(ig, :J)[vmap, vmap]

    set_prop!(sub_ig, :rank, rank)
    set_prop!(sub_ig, :J, J)
    set_prop!(sub_ig, :h, h)
    set_prop!(sub_ig, :vmap, vmap)

    sub_ig
end

function inter_cluster_edges(ig::MetaGraph, cl1::MetaGraph, cl2::MetaGraph)
    verts1, verts2 = get_prop(cl1, :vmap), get_prop(cl2, :vmap)
    outer_edges = filter_edges(
        ig,
        (_, e) -> (src(e) ∈ verts1 && dst(e) ∈ verts2) ||
            (src(e) ∈ verts1 && dst(e) ∈ verts2)
    )
    J = zeros(nv(cl1), nv(cl2))
    # FIXME: don't use indexin
    for e ∈ outer_edges
        @inbounds J[indexin(src(e), verts1)[1], indexin(dst(e), verts2)[1]] = get_prop(ig, e, :J)
    end
    outer_edges, J
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
energy(σ::Vector, ig::MetaGraph) = energy(σ, get_prop(ig, :J)) + energy(σ, get_prop(ig, :h))
