export NetworkGraph
export generate_boundary, generate_tensor
export local_energy, bond_energy

mutable struct NetworkGraph
    factor_graph::MetaDiGraph
    nbrs::Dict
    β::Number

    function NetworkGraph(fg::MetaDiGraph, nbrs::Dict, β::Number)
        ng = new(fg, nbrs, β)

        count = 0
        for v ∈ vertices(fg), w ∈ ng.nbrs[v]
            if has_edge(fg, v, w) count += 1 end
        end

        mc = ne(fg)
        if count < mc
            error("Factor and Ising graphs are incompatible. Edges: $(count) vs $(mc).")
        end
        ng
    end
end

function _get_projector(
    fg::MetaDiGraph,
    v::Int,
    pivot::Int,
    )
    if has_edge(fg, pivot, v)
        _, _, pv = get_prop(fg, pivot, v, :split)
        return pv'
    elseif has_edge(fg, v, pivot)
        pv, _, _ = get_prop(fg, v, pivot, :split)
        return pv
    else
        return nothing
    end
end

@memoize function generate_tensor(ng::NetworkGraph, v::Int)
    fg = ng.factor_graph
    loc_exp = exp.(-ng.β .* get_prop(fg, v, :loc_en))

    dim = []
    @cast tensor[_, i] := loc_exp[i]

    for w ∈ ng.nbrs[v]
        pv = _get_projector(fg, v, w)
        if pv === nothing
            pv = ones(length(loc_exp), 1)
        end
        @cast tensor[(c, γ), σ] |= tensor[c, σ] * pv[σ, γ]
        push!(dim, size(pv, 2))
    end
    reshape(tensor, dim..., :)
end

@memoize function generate_tensor(ng::NetworkGraph, v::Int, w::Int)
    fg = ng.factor_graph
    if has_edge(fg, w, v)
        _, e, _ = get_prop(fg, w, v, :split)
        return exp.(-ng.β .* (e' .- minimum(e)))
    elseif has_edge(fg, v, w)
        _, e, _ = get_prop(fg, v, w, :split)
        return exp.(-ng.β .* (e .- minimum(e)))
    else
        return ones(1, 1)
    end
end


function generate_boundary(
     ng::NetworkGraph,
     v::Int,
     pivot::Int,
     σ::Int
     )
     fg = ng.factor_graph
     if v ∉ vertices(fg) return 1 end

     pv = _get_projector(fg, v, pivot)
     if pv !== nothing
         return findfirst(η -> η > 0, pv[σ, :])
     else
         return 1
     end
 end

 #TODO: this can probably be done better
function bond_energy(
    ng::NetworkGraph,
    u::Int,
    v::Int,
    σ::Int,
    )
    fg = ng.factor_graph
    if has_edge(fg, u, v)
        pu, en, pv = get_prop(fg, u, v, :split)
        energies = (pu * (en * pv[:, σ:σ]))'
    elseif has_edge(fg, v, u)
        pv, en, pu = get_prop(fg, v, u, :split)
        energies = (pv[σ:σ, :] * en) * pu
    else
        energies = zeros(get_prop(fg, u, :loc_dim))
    end
    vec(energies)
end

function local_energy(
    ng::NetworkGraph,
    v::Int,
    )
    get_prop(ng.factor_graph, v, :loc_en)
end
