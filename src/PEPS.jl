export NetworkGraph, PepsNetwork
export generate_tensor, MPO
                              
mutable struct NetworkGraph
    factor_graph::MetaDiGraph
    nbrs::Dict
    β::Number

    function NetworkGraph(factor_graph::MetaDiGraph, nbrs::Dict, β::Number)
        ng = new(factor_graph, nbrs, β) 

        count = 0
        for v ∈ vertices(ng.factor_graph), w ∈ ng.nbrs[v]
            if has_edge(ng.factor_graph, v, w) count += 1 end
        end

        mc = ne(ng.factor_graph)
        if count < mc
            error("Error: $(count) < $(mc)") 
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

function _generate_tensor(ng::NetworkGraph, v::Int)
    fg = ng.factor_graph
    loc_exp = exp.(-ng.β .* get_prop(fg, v, :loc_en))

    projs = Dict()
    for (i, w) ∈ enumerate(ng.nbrs[v])    
        if has_edge(fg, w, v)
            _, _, pv = get_prop(fg, w, v, :split)
            pv = pv'
        elseif has_edge(fg, v, w)
            pv, _, _ = get_prop(fg, v, w, :split)
        else 
            pv = ones(length(loc_exp), 1)
        end
        push!(projs, i => pv)
    end

    L, U, R, D = projs[1], projs[2], projs[3], projs[4]
    @cast tensor[l, u, r, d, σ] |= L[σ, l] * U[σ, u] * R[σ, r] * D[σ, d] * loc_exp[σ]

    tensor
end
_generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}) = _generate_tensor(pn.network_graph, pn.map[m])

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

mutable struct PepsNetwork
    size::NTuple{2, Int}
    map::Dict
    network_graph::NetworkGraph
    origin::Symbol
    i_max::Int
    j_max::Int

    function PepsNetwork(m::Int, n::Int, fg::MetaDiGraph, β::Number, origin::Symbol)
        pn = new((m, n)) 
        pn.map, pn.i_max, pn.j_max = LinearIndices(m, n, origin)

        nbrs = Dict()
        for i ∈ 1:m, j ∈ 1:n
            push!(nbrs, 
            pn.map[i, j] => (pn.map[i, j-1], pn.map[i-1, j], 
                             pn.map[i, j+1], pn.map[i+1, j]))
        end
        pn.network_graph = NetworkGraph(fg, nbrs, β)
        pn
    end
end

generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m])
generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}, n::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])

function MPO(peps::PepsNetwork, i::Int; type::DataType=Float64)
    n = peps.j_max
    ψ = MPO(type, n)
    
    for j ∈ 1:n
        A = generate_tensor(peps, (i, j))
        @reduce B[l, u, r ,d] |= sum(σ) A[l, u, r, d, σ]
        ψ[j] = B
    end

    for j ∈ 1:n-1
        ten = generate_tensor(peps, (i, j), (i, j+1))
        A = ψ[j]
        @tensor B[l, u, r, d] := A[l, u, r̃, d] * ten[r̃, r]
        ψ[j] = B
    end
    ψ
end