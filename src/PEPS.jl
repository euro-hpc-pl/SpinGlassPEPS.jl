export PepsNetwork
export MPO, MPS
export make_lower_MPS

mutable struct PepsNetwork
    size::NTuple{2, Int}
    map::Dict
    network_graph::NetworkGraph
    origin::Symbol
    i_max::Int
    j_max::Int

    function PepsNetwork(m::Int, n::Int, fg::MetaDiGraph, β::Number, origin::Symbol=:NW)
        pn = new((m, n))
        pn.map, pn.i_max, pn.j_max = LinearIndices(m, n, origin)

        nbrs = Dict()
        for i ∈ 1:pn.i_max, j ∈ 1:pn.j_max
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

function MPO(::Type{T}, Ψ::PEPSRow) where {T <: Number}
    n = length(Ψ)
    ϕ = MPO(T, n)
    for i=1:n
        A = Ψ[i]
        @reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        ϕ[i] = B
    end
    ϕ
end
MPO(ψ::PEPSRow) = MPO(Float64, ψ)

function PEPSRow(::Type{T}, peps::PepsNetwork, i::Int) where {T <: Number}
    n = peps.j_max
    ψ = PEPSRow(T, n)

    for j ∈ 1:n ψ[j] = generate_tensor(peps, (i, j)) end

    for j ∈ 2:n
        ten = generate_tensor(peps, (i, j-1), (i, j))
        A = ψ[j]
        @tensor B[l, u, r, d, σ] := ten[l, l̃] * A[l̃, u, r, d, σ] 
        ψ[j] = B
    end
    ψ
end
PEPSRow(peps::PepsNetwork, i::Int) = PEPSRow(Float64, peps, i)

function MPO(::Type{T}, peps::PepsNetwork, i::Int, k::Int) where {T <: Number}
    n = peps.j_max

    ψ = MPO(T, n)
    ng = peps.network_graph
    fg = ng.factor_graph

    for j ∈ 1:n
        v, w = peps.map[i, j], peps.map[k, j]

        if has_edge(fg, v, w)
            _, en, _ = get_prop(fg, v, w, :split)
        elseif has_edge(fg, w, v)
            _, en, _ = get_prop(fg, w, v, :split)
            en = en'
        else
            en = ones(1, 1)
        end

        @cast A[_, u, _, d] |= exp(-ng.β * en[u, d]) 
        ψ[j] = A
    end
    ψ
end
MPO(peps::PepsNetwork, i::Int, k::Int) = MPO(Float64, peps, i, k)

function MPS(::Type{T}, peps::PepsNetwork, i::Int, k::Int) where {T <: Number}
    W = MPO(T, peps, i, k)
    ψ = MPS(T, length(W))
    for (O, i) ∈ enumerate(W) 
        ψ[i] = dropdims(O, dims=(2, 4)) 
    end
    ψ
end
MPS(peps::PepsNetwork, i::Int, k::Int) = MPS(Float64, peps, i, k)

function make_lower_MPS(peps::PepsNetwork, i::Int, k::Int, s::Int, Dcut::Int, tol::Number=1E-8, max_sweeps=4)
    ψ = MPO(PEPSRow(peps, 1))
    #ψ = MPS(peps, i, k)

    for i ∈ s:peps.i_max

        R = PEPSRow(peps, i)
        W = MPO(R)
        M = MPO(peps, i-1, i)

        ψ = (ψ * M) * W

        if (tol > 0.) & (Dcut < size(ψ[1], 3))
            ψ = compress(ψ, Dcut, tol, max_sweeps)
        end
    end
    ψ
end