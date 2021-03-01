export factor_graph, rank_reveal, projectors, split_into_clusters

_max_cell_num(ig::MetaGraph) = maximum(get_prop.(Ref(ig), vertices(ig), :cell))

function split_into_clusters(vertices, assignment_rule)
    # TODO: check how to do this in functional-style
    clusters = Dict(
        i => Set{Int}() for i in values(assignment_rule)
    )
    for v in vertices
        push!(clusters[assignment_rule[v]], v)
    end
    clusters
end

function factor_graph(
    ig::MetaGraph,
    num_states_cl::Int;
    energy::Function=energy,
    spectrum::Function=full_spectrum
    )
    d = _max_cell_num(ig)
    ns = Dict(enumerate(fill(num_states_cl)))
    factor_graph(
        ig,
        ns,
        energy=energy,
        spectrum=spectrum
    )
end

function factor_graph(
    ig::MetaGraph,
    num_states_cl::Dict{Int, Int}=Dict{Int, Int}();
    energy::Function=energy,
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, Int} # e.g. square lattice
)
    L = maximum(values(cluster_assignment_rule))

    fg = MetaDiGraph(L, 0.0)

    cluster_to_verts = split_into_clusters(vertices(ig), cluster_assignment_rule)

    for (v, verts) ∈ cluster_to_verts
        cl = Cluster(ig, verts)
        set_prop!(fg, v, :cluster, cl)
        r = prod(cl.rank)
        num_states = get(num_states_cl, v, r)
        sp = spectrum(cl, num_states=num_states)
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

            set_prop!(fg, e, :split, (pl, en, pr))
        end
    end
    fg
end

function projectors(fg::MetaDiGraph, i::Int, j::Int)
    if has_edge(fg, i, j)
        p1, en, p2 = get_prop(fg, i, j, :split)
    elseif has_edge(fg, j, i)
        p2, en, p1 = get_prop(fg, j, i, :split)
    else
        p1 = en = p2 = ones(1, 1)
    end
    p1, en, p2
end

function rank_reveal(energy, order=:PE)
    @assert order ∈ (:PE, :EP)
    dim = order == :PE ? 1 : 2

    E, idx = unique_dims(energy, dim)

    if order == :PE
        P = zeros(size(energy, 1), size(E, 1))
    else
        P = zeros(size(E, 2), size(energy, 2))
    end

    for (i, elements) ∈ enumerate(eachslice(P, dims=dim))
        elements[idx[i]] = 1
    end

    order == :PE ? (P, E) : (E, P)
end
