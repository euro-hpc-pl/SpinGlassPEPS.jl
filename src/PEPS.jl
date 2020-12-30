export generate_tensor
export NetworkGraph
export PepsNetwork

mutable struct NetworkGraph
    β::Number
    graph::MetaGraph
    nbrs::Dict

    function NetworkGraph(graph::MetaGraph, nbrs::Dict, β::Number)
        ng = new(graph, nbrs, β) 
        count = 0
        for v ∈ vertices(ng.graph), w ∈ ng.nbrs[v]
            if has_edge(ng.graph, v, w)
                count += 1 
            end
        end
        if count < nv(ng.graph)
            error("Error!") 
        end
    end
end


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

#=
function MPO(fg::MetaDiGraph, dim::Symbol=:r, i::Int; T::DataType=Float64)
    @assert dir ∈ (:r, :c)

    m, n = size(fg)
    idx = LinearIndices((1:m, 1:n))
    chain = dim == :r ? fg[idx[:, i]] : fg[idx[i, :]] 

    ψ = MPO(T, length(chain))

    for (j, v) ∈ enumerate(chain)
        ψ[j] = PepsTensor(fg, v).tensor
    end
    ψ
end

function MPS(fg::MetaDiGraph, which::Symbol=:d; T::DataType=Float64)
    @assert which ∈ (:l, :r, :u, :d)

    #ϕ = MPO()

    for (j, v) ∈ enumerate(_row(fg, 1))
        ψ[j] = dropdims(PepsTensor(fg, v).tensor, dims=4)
    end

    # TBW 

    ψ
end
=#

function _linear(m::Int, n::Int)
    map = LinearIndices((1:m, 1:n))
    out = Dict()
    for i ∈ 0:m+1, j ∈ 0:n+1
        try
            push!(out, (i, j) => map[i, j])
        catch
            push!(out, (i, j) => 0)
        end
    end
    out
end

mutable struct PepsNetwork
    m::Int
    n::Int
    β::Number
    map::Dict
    network_graph::MetaGraph
    orientation::Symbol

    function PepsNetwork(m::Int, n::Int, fg::MetaGraph, β::Number)
        pn = new(m, n, β) 
        pn.map = _linear(m, n)
        nbrs = Dict()

        for i ∈ 1:m, j ∈ 1:n
            push!(nbrs, pn.map[i, j] => (pn.map[i-1, j], pn.map[i, j+1], pn.map[i+1, j], pn.map[i, j-1]))
        end

        pn.network_graph = NetworkGraph(fg, nbrs, pn.β)
        pn
    end
end

generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m])
generate_tensor(pn::PepsNetwork, m::NTuple{2,Int}, n::NTuple{2,Int}) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])