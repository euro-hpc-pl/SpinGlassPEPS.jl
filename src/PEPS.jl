export PepsNetwork, contract
export MPO, MPS, boundaryMPS

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
            # v => (l, u, r, d)
            push!(nbrs,
            pn.map[i, j] => (pn.map[i, j-1], pn.map[i-1, j],
                             pn.map[i, j+1], pn.map[i+1, j]))
        end
        pn.network_graph = NetworkGraph(fg, nbrs, β)
        pn
    end
end

generate_tensor(pn::PepsNetwork,
                m::NTuple{2,Int},
                ) = generate_tensor(pn.network_graph, pn.map[m])

generate_tensor(pn::PepsNetwork,
                m::NTuple{2,Int},
                n::NTuple{2,Int},
                ) = generate_tensor(pn.network_graph, pn.map[m], pn.map[n])

function PEPSRow(::Type{T}, peps::PepsNetwork, i::Int) where {T <: Number}
    ψ = PEPSRow(T, peps.j_max)

    # generate tensors from projectors
    for j ∈ 1:length(ψ)
        ψ[j] = generate_tensor(peps, (i, j))
    end

    # include energy
    for j ∈ 1:peps.j_max
        A = ψ[j]
        h = generate_tensor(peps, (i, j-1), (i, j))
        v = generate_tensor(peps, (i-1, j), (i, j))
        @tensor B[l, u, r, d, σ] := h[l, l̃] * v[u, ũ] * A[l̃, ũ, r, d, σ]
        ψ[j] = B
    end
    ψ
end
PEPSRow(peps::PepsNetwork, i::Int) = PEPSRow(Float64, peps, i)

function MPO(::Type{T},
    peps::PepsNetwork,
    i::Int,
    config::Dict{Int, Int} = Dict{Int, Int}()
    ) where {T <: Number}

    W = MPO(T, peps.j_max)
    for (j, A) ∈ enumerate(PEPSRow(peps, i))
        v = get(config, j + peps.j_max * (i - 1), nothing)
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

    function boundaryMPS(
        peps::PepsNetwork,
        range::Int=1,
        Dcut::Int=typemax(Int),
        tol::Number=1E-8,
        max_sweeps=4;
        reversed::Bool=true
        )

        vec = []
        ψ = idMPS(peps.j_max)
        push!(vec, ψ)

        for i ∈ peps.i_max:-1:range
            ψ = MPO(eltype(ψ), peps, i) * ψ
            if bond_dimension(ψ) > Dcut
                ψ = compress(ψ, Dcut, tol, max_sweeps)
            end
            push!(vec, ψ)
        end
        if reversed reverse(vec) else vec end
    end

function LightGraphs.contract(
    peps::PepsNetwork,
    config::Dict{Int, Int} = Dict{Int, Int}(),
    Dcut::Int=typemax(Int),
    tol::Number=1E-8,
    max_sweeps=4,
    )

    ψ = idMPS(peps.j_max)
    for i ∈ peps.i_max:-1:1
        ψ = MPO(eltype(ψ), peps, i, config) * ψ
        if bond_dimension(ψ) > Dcut
            ψ = compress(ψ, Dcut, tol, max_sweeps)
        end
    end
    prod(dropdims(ψ))[]
end


function conditional_pdo(
    peps::PepsNetwork,
    v::Int,
    ∂v::Dict{Int, Int},
    args::Dict,
    )

end
