export NetworkGraph, PepsNetwork
export generate_tensor, MPO

mutable struct NetworkGraph
    factor_graph::MetaGraph
    nbrs::Dict
    β::Number

    function NetworkGraph(factor_graph::MetaGraph, nbrs::Dict, β::Number)
        ng = new(factor_graph, nbrs, β) 

        count = 0
        for v ∈ vertices(ng.factor_graph), w ∈ ng.nbrs[v]
            if has_edge(ng.factor_graph, v, w) count += 1 end
        end

        mc = nv(ng.factor_graph)
        if count < mc
            error("Error: $(count) < $(mc)") 
        end
        ng
    end
end

function generate_tensor(ng::NetworkGraph, v::Int)
    loc_en = get_prop(ng.factor_graph, v, :loc_en)
    tensor = exp.(-ng.β .* loc_en)

    for w ∈ ng.nbrs[v]
        n = max(1, ndims(tensor)-1)
        s = :(@ntuple $n i)

        if has_edge(ng.factor_graph, w, v)
            pw, e, pv = get_prop(ng.factor_graph, w, v, :decomposition)
            @eval @cast tensor[σ, s, γ] |= tensor[σ, s] * pv[γ, σ]

        elseif has_edge(ng.factor_graph, v, w)
            pv, e, pw = get_prop(ng.factor_graph, v, w, :decomposition)
            @eval @cast tensor[σ, s, γ] |= tensor[σ, s] * pv[σ, γ]
        else 
            pv = ones(size(loc_en)...)
            @eval @cast tensor[σ, s, γ] |= tensor[σ, s] * pv[σ, γ]
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

mutable struct PepsNetwork
    size::NTuple{2, Int}
    β::Number
    map::Dict
    network_graph::NetworkGraph
    orientation::Symbol

    function PepsNetwork(m::Int, n::Int, fg::MetaGraph, β::Number)
        pn = new((m, n), β) 
        pn.map = LinearIndices(m, n)

        nbrs = Dict()
        for i ∈ 1:m, j ∈ 1:n
            push!(nbrs, 
            pn.map[i, j] => (pn.map[i-1, j], pn.map[i, j+1], 
                             pn.map[i+1, j], pn.map[i, j-1]))
        end
        pn.network_graph = NetworkGraph(fg, nbrs, pn.β)
        pn
    end
end

generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m])
generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}, n::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])

function _MPO_row(peps::PepsNetwork, i::Int; type::DataType=Float64)
    _, n = peps.size
    ψ = MPO(type, n)
    
    for j ∈ 1:n
        A = generate_tensor(peps, (i, j))
        @reduce B[l, r, u ,d] := sum(σ) A[l, r, u, d, σ]

        # multiply by en
        ψ[j] = B
    end
    ψ
end

function _MPO_column(peps::PepsNetwork, j::Int; type::DataType=Float64)
    m, _ = peps.size
    ψ = MPO(type, m)

    for i ∈ 1:m
        A = generate_tensor(peps, (i, j))
        @reduce B[l, r, u, d] := sum(σ) A[l, r, u, d, σ]

        # multiply by en
        ψ[i] = B
    end
    ψ
end

MPO(peps::PepsNetwork, dim::Symbol, k::Int; type::DataType=Float64) = MPO(peps, Val(dim), k; type=T)
MPO(peps::PepsNetwork, ::Val{:row}, i::Int; type::DataType=Float64) = _MPO_row(peps, i, type=T)
MPO(peps::PepsNetwork, ::Val{:col}, j::Int; type::DataType=Float64) = _MPO_column(peps, j, type=T)

