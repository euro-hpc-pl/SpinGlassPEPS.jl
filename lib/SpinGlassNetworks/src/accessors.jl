# accessors.jl: the single place that knows how model data is stored.
#
# Every property READ in the stack goes through these functions; the writes
# happen only in the graph builders (ising_graph, potts_hamiltonian, the
# truncation/merge constructors). This makes the MetaGraphs Dict{Symbol,Any}
# property bag an implementation detail: the next step of the refactor swaps
# it for a typed struct behind these same signatures without touching any
# call site again.

export projector_pool,
    cluster_graph,
    cluster_spectrum,
    interaction,
    left_projector,
    right_projector,
    outer_cluster_edges,
    bias,
    coupling,
    vertex_rank,
    rank_map,
    grid_size

projector_pool(potts_h::LabelledGraph) =
    get_prop(potts_h, :pool_of_projectors)::PoolOfProjectors{Int}

cluster_graph(potts_h::LabelledGraph, v) = get_prop(potts_h, v, :cluster)

cluster_spectrum(potts_h::LabelledGraph, v) = get_prop(potts_h, v, :spectrum)::Spectrum

interaction(potts_h::LabelledGraph, u, v) = get_prop(potts_h, u, v, :en)

left_projector(potts_h::LabelledGraph, u, v) = get_prop(potts_h, u, v, :ipl)::Int

right_projector(potts_h::LabelledGraph, u, v) = get_prop(potts_h, u, v, :ipr)::Int

outer_cluster_edges(potts_h::LabelledGraph, u, v) = get_prop(potts_h, u, v, :outer_edges)

bias(ig::IsingGraph{T}, v) where {T} = get_prop(ig, v, :h)::T
bias(ig::LabelledGraph, v) = get_prop(ig, v, :h)

coupling(ig::IsingGraph{T}, u, v) where {T} = get_prop(ig, u, v, :J)::T
coupling(ig::LabelledGraph, u, v) = get_prop(ig, u, v, :J)
coupling(ig::IsingGraph{T}, e) where {T} = get_prop(ig, e, :J)::T
coupling(ig::LabelledGraph, e) = get_prop(ig, e, :J)

vertex_rank(ig::LabelledGraph, v) = get_prop(ig, v, :rank)::Int

rank_map(ig::LabelledGraph) = get_prop(ig, :rank)

grid_size(potts_h::LabelledGraph) = (get_prop(potts_h, :Nx)::Int, get_prop(potts_h, :Ny)::Int)
