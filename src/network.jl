export generate_boundary, generate_tensor


function _get_projector(
    fg::MetaDiGraph,
    v::Int,
    w::Int
    )
    if has_edge(fg, w, v)
        get_prop(fg, w, v, :pr)'
    elseif has_edge(fg, v, w)
        get_prop(fg, v, w, :pl)
    else
        loc_dim = length(get_prop(fg, v, :loc_en))
        ones(loc_dim, 1)
    end
end

@memoize function generate_tensor(network::PEPSNetwork, v::Int)
    loc_exp = exp.(-network.β .* get_prop(network.fg, v, :loc_en))

    dim = zeros(Int, length(network.nbrs[v]))
    @cast A[_, i] := loc_exp[i]

    for (j, w) ∈ enumerate(network.nbrs[v])
        pv = _get_projector(network.fg, v, w)
        @cast A[(c, γ), σ] |= A[c, σ] * pv[σ, γ]
        dim[j] = size(pv, 2)
    end
    reshape(A, dim..., :)
end

@memoize function generate_tensor(network::PEPSNetwork, v::Int, w::Int)
    fg = network.fg
    if has_edge(fg, w, v)
        en = get_prop(fg, w, v, :en)'
    elseif has_edge(fg, v, w)
        en = get_prop(fg, v, w, :en)
    else
        en = zeros(1, 1)
    end
    exp.(-network.β .* en)
end

function generate_boundary(
    fg::MetaDiGraph,
    v::Int,
    w::Int,
    state::Int
    )
    if v ∉ vertices(fg) return 1 end
    loc_dim = length(get_prop(fg, v, :loc_en))
    pv = _get_projector(fg, v, w)
    findfirst(x -> x > 0, pv[state, :])
end
