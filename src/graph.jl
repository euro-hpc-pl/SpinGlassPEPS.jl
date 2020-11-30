export Edge, Node, Grid

const Node = Int
const Edge = Tuple{Node, Node}

const IsingInstance = Dict{Edge, Float64}

struct Cluster
    instance::IsingInstance
    nodes::Vector{Node}
    legs::Dict{Symbol, Dict{Node, Vector{Node}}}
end

export chimera_cell
struct Chimera
    size::NTuple
    graph::MetaGraph
end

function chimera(chimera::Chimera)
    N, M, _ = chimera.size
    linear = LinearIndices((1:N, 1:M))

    for v ∈ vertices(chimera.graph)
        i, j, u, k = linear_to_chimera(v, chimera.size)

        set_prop!(chimera.graph, v, :chimera_index, (i, j, u, k))
        x = linear[i, j]
        set_prop!(chimera.graph, v, :cell_index, x)

        for w ∈ unique_neighbors(chimera.graph, v)
            ĩ, j̃, _, _ = linear_to_chimera(w, chimera.size)
            y = linear[ĩ, j̃]
            set_prop!(chimera.graph, v, w, :cells_edge, (x, y))
        end
    end
end

#=
function energy(chimera::Chimera, σ::State, n::Int)
    energy::Float64 = 0

    edges = filter_edges(chimera.graph, :cells_edge, (n, n))
    for edge ∈ edges
        i, j = src(edge), dst(edge)         
        J = get_prop(ig, i, j, :J) 
        energy += σ[i] * J * σ[j]   
    end 

    vertices = filter_vertices(chimera.graph, :cell, n)
    for i ∈ vertices
        h = get_prop(ig, i, :h)  
        energy += h * σ[i]
    end    
    -energy
end
=#

#=
"""
$(TYPEDSIGNATURES)

Returns n-th unit cell of Chimera graph.

"""
function cell(chimera::Chimera, n::Int)
    N, M, T = chimera.size
    C = N * M

    @assert n <= C "Cell $n is outside the graph C$C."

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
    cell
end

"""
$(TYPEDSIGNATURES)

Returns a graph of interactions between Chimera cells A and B.

"""
function bond(ig::MetaGraph, A::MetaGraph, B::MetaGraph)
    int = zero(ig)

    for i ∈ vertices(A)
        v = get_prop(A, i, :global)            

        set_prop!(int, i, :global, v)
        set_prop!(int, i, :h, 0.)

        nbrs = unique_neighbors(ig, v)

        for w ∈ intersect(vertices(B), nbrs)
            J = get_prop(ig, v, w, :J)
            set_prop!(int, i, j, :J, J)
        end
    end
    int
end

function factor_graph(chimera::Chimera)
    N, M, T = chimera.size
 
    fg = MetaGraph(Grid((N, M)))

    for v ∈ vertices(fg)
        nbrs = unique_neighbors(fg, v)

        if !isempty(nbrs)
            vcell = cell(chimera, v)
            set_prop!(fg, v, :node, cell)

            for w ∈ nbrs
                wcell = cell(chimera, v)
                bond = bond(chimera.graph, vcell, wcell)
                set_prop!(fg, v, w, :bond, bond)
            end
        end
    end
    fg
end
=#