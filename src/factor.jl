export factor_graph
export rank_reveal, projectors

function _max_cell_num(ig::MetaGraph)
    L = 0
    for v ∈ vertices(ig)
        L = max(L, get_prop(ig, v, :cell))
    end
    L
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
    spectrum::Function=full_spectrum
)
    L = _max_cell_num(ig)
    fg = MetaDiGraph(L, 0.0)

    for v ∈ vertices(fg)
        cl = Cluster(ig, v)
        set_prop!(fg, v, :cluster, cl)
        r = prod(cl.rank)
        num_states = get(num_states_cl, v, r)
        sp = spectrum(cl, num_states=num_states)
        set_prop!(fg, v, :spectrum, sp)
        set_prop!(fg, v, :loc_en, vec(sp.energies))
        set_prop!(fg, v, :loc_dim, length(vec(sp.energies)))
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

# function projectors(fg::MetaDiGraph, i::Int, j::Int)
#     if has_edge(fg, i, j)
#         p1, en, p2 = get_prop(fg, i, j, :split)
#     elseif has_edge(fg, j, i)
#         p2, en, p1 = get_prop(fg, j, i, :split)
#     else
#         p1 = en = p2 = ones(1, 1)
#     end
#     p1, en, p2
# end

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
