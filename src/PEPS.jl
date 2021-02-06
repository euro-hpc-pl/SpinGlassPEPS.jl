export PepsNetwork
export MPO, MPS
export boundaryMPS, contract

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
    for j ∈ 1:n 
        ψ[j] = generate_tensor(peps, (i, j)) 
    end

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
            en = zeros(1, 1)
        end

        @cast A[_, u, _, d] |= exp(-ng.β * en[u, d])
        ψ[j] = A
    end
    ψ
end
MPO(peps::PepsNetwork, i::Int, k::Int) = MPO(Float64, peps, i, k)

function _MPS(::Type{T}, peps::PepsNetwork) where {T <: Number}
    W = MPO(PEPSRow(peps, peps.i_max))
    ψ = MPS(T, length(W))

    for (i, O) ∈ enumerate(W)
        ψ[i] = dropdims(O, dims=4)
    end
    ψ
end

function MPS(::Type{T}, peps::PepsNetwork) where {T <: Number}
    ψ = MPS(T, peps.j_max)
    for i ∈ 1:length(ψ)
        ψ[i] = ones(1, 1, 1)
    end
    ψ
end
MPS(peps::PepsNetwork) = MPS(Float64, peps)

function boundaryMPS(peps::PepsNetwork,
    Dcut::Int=typemax(Int),
    tol::Number=1E-8,
    max_sweeps=4)

    MPS_vec = Vector{MPS}(undef, peps.i_max)

    ψ = MPS(peps)
    for i ∈ peps.i_max:-1:1
        W = MPO(PEPSRow(peps, i))
        M = MPO(peps, i, i+1)
        ψ = W * (M * ψ)

        if bond_dimension(ψ) > Dcut
            ψ = compress(ψ, Dcut, tol, max_sweeps)
        end

        MPS_vec[peps.i_max - i + 1] = ψ
    end
    MPS_vec
end

function MPO(::Type{T}, 
    peps::PepsNetwork, 
    i::Int, 
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) where {T <: Number}

    W = MPO(T, peps.j_max)

    for (j, A) ∈ enumerate(PEPSRow(peps, i)) 
        ij = j + peps.j_max * (i - 1)
        v = get(config, ij, nothing)

        if v !== nothing
            @cast B[l, u, r, d] |= A[l, u, r, d, $(v)]
        else
            @reduce B[l, u, r, d] |= sum(σ) A[l, u, r, d, σ]
        end
        W[j] = B
    end
    W 
end
MPO(peps::PepsNetwork, 
    i::Int, 
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) = MPO(Float64, peps, i, config)

function LightGraphs.contract(
    peps::PepsNetwork,
    config::Dict{Int, Int} = Dict{Int, Int}(),
    Dcut::Int=typemax(Int),
    tol::Number=1E-8,
    max_sweeps=4,
    )

    ψ = MPS(peps)
    T = eltype(ψ)

    for i ∈ peps.i_max:-1:1
        W = MPO(T, peps, i, config)
        M = MPO(T, peps, i, i+1)
        ψ = W * (M * ψ)
        if bond_dimension(ψ) > Dcut
            ψ = compress(ψ, Dcut, tol, max_sweeps)
        end
    end

    Z = Array{T, 2}[]
    for A ∈ ψ push!(Z, dropdims(A, dims=2)) end
    prod(Z)[]
end
