struct Chimera
    size::NTuple
    graph::MetaGraph
end

"""
$(TYPEDSIGNATURES)

Returns Chimera's unit cell at \$(i, j)\$ on a grid.

"""
function chimera_cell(chimera::Chimera, i::Int, j::Int)
    N, M, T = chimera.size
    C = N * M

    @assert i <= N & j <= M "Cell ($i, $j) is outside the graph C$C."

    vlist = [ chimera_to_linear(i, j, u, k) for u ∈ 1:T for k ∈ [0, 1] ]
    cell, vmap = induced_subgraph(chimera.graph, vlist)

    for (i, v) ∈ zip(nv(cell), vmap)
        set_prop!(cell, i, :global, v)

        nbrs = unique_neighbors(chimera.graph, v)

        for w ∈ intersect(vlist, nbrs)
            J = get_prop(chimera.graph, v, w, :J)
            set_prop!(cell, i, j, :J, J)
        end

        h = get_prop(chimera.graph, v, :h)
        set_prop!(cell, i, :h, h)
    end
    
    set_prop!(cell, :description, "Unit cell ($i, $j) of Chimera $C.") 
    cell
end

end