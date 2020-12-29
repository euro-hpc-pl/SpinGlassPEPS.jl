export generate_tensor

function generate_tensor(ng::NetworkGraph, v::Int)
    loc_en = get_prop(ng.graph, v, :loc_en)
    tensor = exp.(-ng.β .* loc_en)

    for w ∈ ng.nbrs[v]
        n = max(1, ndims(tensor)-1)
        s = :(@ntuple n i)

        if has_edge(ng.graph, w, v)
            pw, e, pv = get_prop(ng.graph, w, v, :decomposition)
            @eval @cast tensor[σ, s..., γ] |= tensor[σ, s...] * pv[γ, σ]

        elseif has_edge(ng.graph, v, w)
            pv, e, pw = get_prop(ng.graph, v, w, :decomposition)
            @eval @cast tensor[σ, s..., γ] |= tensor[σ, s...] * pv[σ, γ]
        else 
            pv = ones(size(loc_en), 1)
            @eval @cast tensor[σ, s..., γ] |= tensor[σ, s...] * pv[σ, γ]
        end
    end
    tensor
end

function generate_tensor(ng::NetworkGraph, v::Int, w::Int)
    if has_edge(ng.graph, w, v)
        _, e, _ = get_prop(ng.graph, w, v, :decomposition)
        tensor = exp.(-ng.β .* e') 
    elseif has_edge(ng.graph, v, w)
        _, e, _ = get_prop(ng.graph, v, w, :decomposition)
        tensor = exp.(-ng.β .* e) 
    else 
        tensor = ones(1, 1)
    end
    tensor
end
