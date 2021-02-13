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
            error("Factor and Ising graphs are incompatible. Edges: $(count) vs $(mc).") 
        end
        ng
    end
end

@memoize function generate_tensor(ng::NetworkGraph, v::Int)
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

@memoize function generate_tensor(ng::NetworkGraph, v::Int, w::Int)
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

function generate_boundary(
    ng::NetworkGraph, 
    v::Int, 
    w::Int, 
    state::Int
    )
    fg = ng.factor_graph
    if v ∉ vertices(fg) return 1 end
    loc_dim = length(get_prop(fg, v, :loc_en))
    if has_edge(fg, w, v)
        _, _, pv = get_prop(fg, w, v, :split)
        pv = pv'
    elseif has_edge(fg, v, w)
        pv, _, _ = get_prop(fg, v, w, :split)
    else
        pv = ones(loc_dim, 1)
    end

    v = pv[state, :]
    findfirst(x -> x>0, v)
    #=
    position = 1
    for i ∈ 1:length(v)
        if v[i] > 0 position = i end
    end
    position=#
end