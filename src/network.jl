export NetworkGraph, generate_tensor

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
            error("Factor and Ising graphs are incompatible: $(count) < $(mc)") 
        end
        ng
    end
end

function generate_tensor(ng::NetworkGraph, v::Int)
    fg = ng.factor_graph
    loc_exp = exp.(-ng.β .* get_prop(fg, v, :loc_en))

    dim = []
    @cast tensor[_, i] := loc_exp[i]

    for w ∈ ng.nbrs[v]
        if has_edge(fg, w, v)
            _, _, pv = get_prop(fg, w, v, :split)
            pv = pv'
        elseif has_edge(fg, v, w)
            pv, _, _ = get_prop(fg, v, w, :split)
        else
            pv = ones(length(loc_exp), 1)
        end

        @cast tensor[(c, γ), σ] |= tensor[c, σ] * pv[σ, γ]
        push!(dim, size(pv, 2))
    end
    reshape(tensor, dim..., :)
end

function generate_tensor(ng::NetworkGraph, v::Int, w::Int)
    fg = ng.factor_graph
    if has_edge(fg, w, v)
        _, e, _ = get_prop(fg, w, v, :split)
        tensor = exp.(-ng.β .* e')
    elseif has_edge(fg, v, w)
        _, e, _ = get_prop(fg, v, w, :split)
        tensor = exp.(-ng.β .* e)
    else
        tensor = ones(1, 1)
    end
    tensor
end