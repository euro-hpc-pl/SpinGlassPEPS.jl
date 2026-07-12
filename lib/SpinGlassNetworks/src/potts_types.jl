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

struct PottsHamiltonian{L,T<:Real,C,SP<:Spectrum,OE}
    topology::LabelledGraph{SimpleDiGraph{Int},L}
    clusters::Dict{L,C}
    spectra::Dict{L,SP}
    interactions::Dict{Tuple{L,L},ClusterInteraction{T,OE}}
    pool::PoolOfProjectors{Int}
end

# `potts_h.reverse_label_map` / `.labels` are used by downstream consumers;
# forward them to the topology.
function Base.getproperty(ph::PottsHamiltonian, s::Symbol)
    s ∈ (:labels, :reverse_label_map) && return getproperty(getfield(ph, :topology), s)
    getfield(ph, s)
end

# --- graph interface, delegated to the topology ---------------------------
Graphs.vertices(ph::PottsHamiltonian) = vertices(ph.topology)
Graphs.edges(ph::PottsHamiltonian) = edges(ph.topology)
Graphs.nv(ph::PottsHamiltonian) = nv(ph.topology)
Graphs.ne(ph::PottsHamiltonian) = ne(ph.topology)
Graphs.has_vertex(ph::PottsHamiltonian, v) = has_vertex(ph.topology, v)
Graphs.has_edge(ph::PottsHamiltonian, u, v) = has_edge(ph.topology, u, v)
Graphs.neighbors(ph::PottsHamiltonian, v) = neighbors(ph.topology, v)
Graphs.inneighbors(ph::PottsHamiltonian, v) = inneighbors(ph.topology, v)
Graphs.outneighbors(ph::PottsHamiltonian, v) = outneighbors(ph.topology, v)

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
