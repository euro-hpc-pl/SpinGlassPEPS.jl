export factor_graph, rank_reveal, projectors, split_into_clusters


function split_into_clusters(vertices, assignment_rule)
    # TODO: check how to do this in functional-style
    clusters = Dict(
        i => [] for i in values(assignment_rule)
    )
    for v in vertices
        push!(clusters[assignment_rule[v]], v)
    end
    clusters
end

function split_into_clusters(ig::MetaGraph, assignment_rule)
    cluster_id_to_verts = Dict(
        i => Int[] for i in values(assignment_rule)
    )

    for (i, v) in enumerate(nodes(ig))
        push!(cluster_id_to_verts[assignment_rule[v]], i)
    end

    return Dict(
        i => cluster(ig, verts) for (i, verts) ∈ cluster_id_to_verts
    )
end

function factor_graph(
    ig::MetaGraph,
    num_states_cl::Int;
    energy::Function=energy,
    spectrum::Function=full_spectrum,
    cluster_assignment_rule::Dict{Int, Int} # e.g. square lattice
)
    ns = Dict(i => num_states_cl for i ∈ Set(values(cluster_assignment_rule)))
    factor_graph(
        ig,
        ns,
        energy=energy,
        spectrum=spectrum,
        cluster_assignment_rule=cluster_assignment_rule
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
    fg = MetaDiGraph(L)

    for (v, cl) ∈ split_into_clusters(ig, cluster_assignment_rule)
        set_prop!(fg, v, :cluster, cl)
        sp = spectrum(cl, num_states=get(num_states_cl, v, basis_size(cl)))
        set_prop!(fg, v, :spectrum, sp)
        set_prop!(fg, v, :loc_en, vec(sp.energies))
    end

    for i ∈ 1:L, j ∈ i+1:L
        v, w = get_prop(fg, i, :cluster), get_prop(fg, j, :cluster)

        outer_edges, J = inter_cluster_edges(ig, v, w)

        if !isempty(outer_edges)
            en = inter_cluster_energy(
                get_prop(fg, i, :spectrum).states, J, get_prop(fg, j, :spectrum).states
            )

            pl, en = rank_reveal(en, :PE)
            en, pr = rank_reveal(en, :EP)

            add_edge!(
                fg, i, j,
                Dict(:outer_edges => outer_edges, :pl => pl, :en => en, :pr => pr)
            )
        end
    end
    fg
end


# TODO: eradicate it
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
