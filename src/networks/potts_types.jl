# potts_types.jl: the typed Potts Hamiltonian (standard flavor).
#
# The standard clustered Hamiltonian - what the PEPS solver consumes - stores
# its data in concrete typed fields instead of MetaGraphs Dict{Symbol,Any}
# properties. The legacy LabelledGraph representation remains in use for the
# 2-site belief-propagation graph and the RMF loader; both representations
# answer to the same accessor functions, so consumers cannot tell them apart.

export PottsHamiltonian, PottsLike

struct ClusterInteraction{T<:Real,OE}
    en::Matrix{T}
    ipl::Int
    ipr::Int
    outer_edges::OE
end

struct PottsHamiltonian{L,T<:Real,C,SP<:Spectrum,OE} <: AbstractGraph{L}
    topology::LabelledGraph{SimpleDiGraph{Int},L}
    clusters::Dict{L,C}
    spectra::Dict{L,SP}
    interactions::Dict{Tuple{L,L},ClusterInteraction{T,OE}}
    pool::PoolOfProjectors{Int}
end

# Preserve the public graph properties of the former LabelledGraph
# representation.  The typed storage remains available through `topology`.
function Base.getproperty(ph::PottsHamiltonian, s::Symbol)
    s ∈ (:labels, :reverse_label_map, :inner_graph) &&
        return getproperty(getfield(ph, :topology), s)
    getfield(ph, s)
end

function Base.propertynames(ph::PottsHamiltonian, private::Bool = false)
    (fieldnames(typeof(ph))..., :labels, :reverse_label_map, :inner_graph)
end

# --- graph interface, delegated to the topology ---------------------------
Graphs.vertices(ph::PottsHamiltonian) = vertices(ph.topology)
Graphs.edges(ph::PottsHamiltonian) = edges(ph.topology)
Graphs.nv(ph::PottsHamiltonian) = nv(ph.topology)
Graphs.ne(ph::PottsHamiltonian) = ne(ph.topology)
Graphs.has_vertex(ph::PottsHamiltonian, v) = has_vertex(ph.topology, v)
Graphs.has_edge(ph::PottsHamiltonian, u, v) = has_edge(ph.topology, u, v)
Graphs.has_edge(ph::PottsHamiltonian, e::Graphs.AbstractEdge) = has_edge(ph, src(e), dst(e))
Graphs.neighbors(ph::PottsHamiltonian{L}, v::L) where {L} = outneighbors(ph.topology, v)
Graphs.inneighbors(ph::PottsHamiltonian{L}, v::L) where {L} = inneighbors(ph.topology, v)
Graphs.outneighbors(ph::PottsHamiltonian{L}, v::L) where {L} = outneighbors(ph.topology, v)
Graphs.all_neighbors(ph::PottsHamiltonian{L}, v::L) where {L} = all_neighbors(ph.topology, v)
Graphs.is_directed(::Type{<:PottsHamiltonian}) = true
Graphs.edgetype(::PottsHamiltonian{L}) where {L} = LabelledEdge{L}

# Graphs.jl's generic degree methods require integer vertex labels. Potts
# vertices are commonly coordinate tuples, so define the scalar and collection
# forms explicitly.
Graphs.indegree(ph::PottsHamiltonian{L}, v::L) where {L} = length(inneighbors(ph, v))
Graphs.indegree(ph::PottsHamiltonian) = indegree(ph, vertices(ph))
Graphs.indegree(ph::PottsHamiltonian{L}, vs::AbstractVector{L}) where {L} =
    [indegree(ph, v) for v in vs]
Graphs.outdegree(ph::PottsHamiltonian{L}, v::L) where {L} = length(outneighbors(ph, v))
Graphs.outdegree(ph::PottsHamiltonian) = outdegree(ph, vertices(ph))
Graphs.outdegree(ph::PottsHamiltonian{L}, vs::AbstractVector{L}) where {L} =
    [outdegree(ph, v) for v in vs]
Graphs.degree(ph::PottsHamiltonian{L}, v::L) where {L} = indegree(ph, v) + outdegree(ph, v)
Graphs.degree(ph::PottsHamiltonian) = degree(ph, vertices(ph))
Graphs.degree(ph::PottsHamiltonian{L}, vs::AbstractVector{L}) where {L} =
    [degree(ph, v) for v in vs]

# Return the standard adjacency matrix in the topology's label order. Calling
# the Graphs.jl fallback directly would try to use tuple labels as row indices.
function Graphs.adjacency_matrix(
    ph::PottsHamiltonian,
    T::DataType = Int;
    dir::Symbol = :out,
)
    adjacency_matrix(ph.inner_graph, T; dir)
end

# --- accessors -------------------------------------------------------------
projector_pool(ph::PottsHamiltonian) = ph.pool
cluster_graph(ph::PottsHamiltonian, v) = ph.clusters[v]
cluster_spectrum(ph::PottsHamiltonian, v) = ph.spectra[v]
interaction(ph::PottsHamiltonian, u, v) = ph.interactions[(u, v)].en
left_projector(ph::PottsHamiltonian, u, v) = ph.interactions[(u, v)].ipl
right_projector(ph::PottsHamiltonian, u, v) = ph.interactions[(u, v)].ipr
outer_cluster_edges(ph::PottsHamiltonian, u, v) = ph.interactions[(u, v)].outer_edges

# Either representation of a Potts Hamiltonian; consumers accept both while
# the 2-site BP graph and the RMF loader still use the legacy one.
const PottsLike = Union{LabelledGraph,PottsHamiltonian}

# --- MetaGraphs-compatible property access (back-compat) --------------------
# Downstream/user code written against the LabelledGraph representation used
# get_prop directly; keep it working on the typed struct. New code should use
# the accessor functions above.
function MetaGraphs.get_prop(ph::PottsHamiltonian, s::Symbol)
    s === :pool_of_projectors && return ph.pool
    throw(KeyError(s))
end

function MetaGraphs.get_prop(ph::PottsHamiltonian, v, s::Symbol)
    s === :cluster && return cluster_graph(ph, v)
    s === :spectrum && return cluster_spectrum(ph, v)
    throw(KeyError(s))
end

MetaGraphs.get_prop(ph::PottsHamiltonian, e::LabelledEdge, s::Symbol) =
    get_prop(ph, src(e), dst(e), s)

function MetaGraphs.get_prop(ph::PottsHamiltonian, u, v, s::Symbol)
    s === :en && return interaction(ph, u, v)
    s === :ipl && return left_projector(ph, u, v)
    s === :ipr && return right_projector(ph, u, v)
    s === :outer_edges && return outer_cluster_edges(ph, u, v)
    throw(KeyError(s))
end
