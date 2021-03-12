export ising_graph
export energy, rank_vec
export Spectrum, cluster, rank, nodes, basis_size

const Instance = Union{String, Dict}

struct Spectrum
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
end

unique_nodes(ising_tuples) = sort(collect(Set(Iterators.flatten((i, j) for (i, j, _) ∈ ising_tuples))))

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

    original_nodes = unique_nodes(ising)
    L = length(original_nodes)
    nodes_to_vertices = Dict(w => i for (i, w) ∈ enumerate(original_nodes))

    ig = MetaGraph(length(original_nodes))

    foreach(args -> set_prop!(ig, args[1], :node, args[2]), enumerate(original_nodes))

    J = zeros(L, L)
    h = zeros(L)

    # setup the model (J_ij, h_i)
    for (_i, _j, v) ∈ ising
        i, j = nodes_to_vertices[_i], nodes_to_vertices[_j]
        v *= sgn

        if i == j
            h[i] = v
        else
            add_edge!(ig, i, j, :J, v) || throw(ArgumentError("Duplicate Egde ($i, $j)"))
            J[i, j] = v
        end
    end

    foreach(i -> set_prop!(ig, i, :h, h[i]), vertices(ig))

    set_prop!(
        ig,
        :rank,
        Dict{Int, Int}(
            v => get(rank_override, w, 2) for (w, v) in nodes_to_vertices
        )
    )

    set_prop!(ig, :J, J)
    set_prop!(ig, :h, h)
    set_prop!(ig, :nodes_map, nodes_to_vertices)
    ig
end

nodes(ig::MetaGraph) = collect(get_prop.(Ref(ig), vertices(ig), :node))
rank_vec(ig::MetaGraph) = collect(values(get_prop(ig, :rank)))
basis_size(ig::MetaGraph) = prod(prod(rank_vec(ig)))

function cluster(ig::MetaGraph, verts)
    sub_ig, vmap = induced_subgraph(ig, collect(verts))

    h = get_prop.(Ref(sub_ig), vertices(sub_ig), :h)
    rank = getindex.(Ref(get_prop(ig, :rank)), vmap)
    J = get_prop(ig, :J)[vmap, vmap]

    set_props!(sub_ig, Dict(:rank => rank, :J => J, :h => h, :vmap => vmap))
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


# Please don't make the below another energy method.
# There is already so much mess going on :)
function inter_cluster_energy(cl1_states, J::Matrix, cl2_states)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
