using LabelledGraphs

export ising_graph,
    rank_vec,
    cluster,
    rank,
    nodes,
    basis_size,
    biases,
    couplings,
    IsingGraph,
    prune,
    inter_cluster_edges

const Instance = Union{String,Dict}
const IsingGraph{T} = LabelledGraph{MetaGraph{Int,T}}

function unique_nodes(ising_tuples)
    sort(collect(Set(Iterators.flatten((i, j) for (i, j, _) ∈ ising_tuples))))
end

"""
$(TYPEDSIGNATURES)

Create an Ising graph from interaction data.

This function creates an Ising graph (LabelledGraph) from interaction data provided in the form of an `inst` argument.
The Ising graph represents a system of spins, where each spin is associated with a vertex, 
and interactions between spins are represented as edges with corresponding weights.

# Arguments:
- `::Type{T}`: The type of the edge weights, typically `Float64` or `Float32`.
- `inst::Instance`: Interaction data, which can be either a file path to a CSV file or a collection of triples `(i, j, J)` representing interactions between spins, where `i` and `j` are spin indices, and `J` is the interaction strength.
- `scale::Real`: The scale factor establishes the convention in the Hamiltonian (default is 1).
- `rank_override::Dict`: A dictionary specifying the rank (number of states) for each vertex. If not provided, a default rank of 2 is used for all vertices.

# Returns:
- `ig::IsingGraph{T}`: The Ising graph (LabelledGraph) representing the spin system.

The function reads interaction data and constructs an Ising graph `ig`.
It assigns interaction strengths to edges between spins and optionally scales them by the `scale` factor. 'Scale' option allows for the change of convention in the Hamiltonian.
The `rank_override` dictionary can be used to specify the rank (number of states) for individual vertices, allowing customization of the Ising model.
Convention: H = scale * sum_{i, j} (J_{ij} * s_i * s_j + J_{ii} * s_i)
"""
function ising_graph(
    ::Type{T},
    inst::Instance;
    scale::Real = 1,
    rank_override::Dict = Dict{Int,Int}(),
) where {T}
    if inst isa String
        ising = CSV.File(inst, types = [Int, Int, T], header = 0, comment = "#")
    else
        ising = [(i, j, T(J)) for ((i, j), J) ∈ inst]
    end
    ig = IsingGraph{T}(unique_nodes(ising))

    set_prop!.(Ref(ig), vertices(ig), :h, zero(T))
    foreach(v -> set_prop!(ig, v, :rank, get(rank_override, v, 2)), vertices(ig))

    for (i, j, v) ∈ ising
        v *= T(scale)
        if i == j
            set_prop!(ig, i, :h, v)
        else
            add_edge!(ig, i, j) || throw(ArgumentError("Duplicate Egde ($i, $j)"))
            set_prop!(ig, i, j, :J, v)
        end
    end
    set_prop!(ig, :rank, Dict(v => get(rank_override, v, 2) for v in vertices(ig)))
    ig
end

function ising_graph(inst::Instance; scale::Real = 1, rank_override::Dict = Dict{Int,Int}())
    ising_graph(Float64, inst; scale = scale, rank_override = rank_override)
end
Base.eltype(ig::IsingGraph{T}) where {T} = T

rank_vec(ig::IsingGraph) = Int[get_prop((ig), v, :rank) for v ∈ vertices(ig)]
basis_size(ig::IsingGraph) = prod(rank_vec(ig))
biases(ig::IsingGraph) = get_prop.(Ref(ig), vertices(ig), :h)

"""
$(TYPEDSIGNATURES)

Return the coupling strengths between vertices of an Ising graph.

This function computes and returns the coupling strengths (interaction energies) between pairs of vertices in an Ising graph `ig`.
The coupling strengths are represented as a matrix, where each element `(i, j)` corresponds to the interaction energy between vertex `i` and vertex `j`.

# Arguments:
- `ig::IsingGraph{T}`: The Ising graph representing a system of spins with associated interaction strengths.

# Returns:
- `J::Matrix{T}`: A matrix of coupling strengths between vertices of the Ising graph.

The function iterates over the edges of the Ising graph and extracts the interaction strengths associated with each edge, populating the `J` matrix accordingly.
"""
function couplings(ig::IsingGraph{T}) where {T}
    J = zeros(T, nv(ig), nv(ig))
    for edge ∈ edges(ig)
        i = ig.reverse_label_map[src(edge)]
        j = ig.reverse_label_map[dst(edge)]
        @inbounds J[i, j] = get_prop(ig, edge, :J)
    end
    J
end
cluster(ig::IsingGraph, verts) = induced_subgraph(ig, collect(verts))

"""
$(TYPEDSIGNATURES)

Return the dense adjacency matrix between clusters of vertices in an Ising graph.

This function computes and returns the dense adjacency matrix `J` between clusters of vertices represented by two
Ising graphs, `cl1` and `cl2`, within the context of the larger Ising graph `ig`.
The adjacency matrix represents the interaction strengths between clusters of vertices,
where each element `(i, j)` corresponds to the interaction strength between cluster `i` in `cl1` and cluster `j` in `cl2`.

# Arguments:
- `ig::IsingGraph{T}`: The Ising graph representing a system of spins with associated interaction strengths.
- `cl1::IsingGraph{T}`: The first Ising graph representing one cluster of vertices.
- `cl2::IsingGraph{T}`: The second Ising graph representing another cluster of vertices.

# Returns:
- `outer_edges::Vector{LabelledEdge}`: A vector of labeled edges representing the interactions between clusters.
- `J::Matrix{T}`: A dense adjacency matrix representing interaction strengths between clusters.

The function first identifies the outer edges that connect vertices between the two clusters in the context of the larger Ising graph `ig`.
It then computes the interaction strengths associated with these outer edges and populates the dense adjacency matrix `J` accordingly.
"""
function inter_cluster_edges(
    ig::IsingGraph{T},
    cl1::IsingGraph{T},
    cl2::IsingGraph{T},
) where {T}
    outer_edges =
        [LabelledEdge(i, j) for i ∈ vertices(cl1), j ∈ vertices(cl2) if has_edge(ig, i, j)]
    J = zeros(T, nv(cl1), nv(cl2))
    for e ∈ outer_edges
        i, j = cl1.reverse_label_map[src(e)], cl2.reverse_label_map[dst(e)]
        @inbounds J[i, j] = get_prop(ig, e, :J)
    end
    outer_edges, J
end

"""
$(TYPEDSIGNATURES)

Used only in MPS_search, would be obsolete if MPS_search uses QMps.
Remove non-existing spins from an Ising graph.

This function removes non-existing spins from the given Ising graph `ig`.
Non-existing spins are those that have zero degree (no connections to other spins) and also have an external
magnetic field (`h`) that is not approximately equal to zero within the specified tolerance `atol`.

# Arguments:
- `ig::IsingGraph`: The Ising graph to be pruned.
- `atol::Real`: The tolerance for considering the external magnetic field as zero. The default value is `1e-14`.

# Returns:
- `pruned_graph::IsingGraph`: A new Ising graph with non-existing spins removed.

The function returns a pruned version of the input Ising graph, where non-existing spins and their associated properties are removed.
"""
function prune(ig::IsingGraph; atol::Real = 1e-14)
    to_keep = vcat(
        findall(!iszero, degree(ig)),
        findall(
            x ->
                iszero(degree(ig, x)) && !isapprox(get_prop(ig, x, :h), 0, atol = atol),
            vertices(ig),
        ),
    )
    gg = ig[ig.labels[to_keep]]
    labels = collect(vertices(gg.inner_graph))
    reverse_label_map = Dict(i => i for i = 1:nv(gg.inner_graph))
    LabelledGraph(labels, gg.inner_graph, reverse_label_map)
end
