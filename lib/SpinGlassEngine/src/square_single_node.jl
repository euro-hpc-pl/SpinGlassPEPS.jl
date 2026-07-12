
export site,
    SquareSingleNode,
    tensor_map,
    gauges_list,
    MpoLayers,
    conditional_probability,
    projectors_site_tensor,
    nodes_search_order_Mps,
    boundary,
    update_energy,
    projected_energy


"""
$(TYPEDSIGNATURES)

A geometric structure representing a 1-layer grid with nodes arranged in a grid of rows and columns. 

# Type Parameters
- `T <: AbstractTensorsLayout`: The layout of decomposition of tensors into MPS. Can be `GaugesEnergy`, `EnergyGauges` or `EngGaugesEng`.

# Constructors
- `SquareDoubleNode(layout::T)`: Create a `SquareDoubleNode` with the specified tensor layout.
"""
struct SquareSingleNode{T<:AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)

Creates SquareSingleNode geometry as a LabelledGraph.
Create a labelled grid graph with nodes arranged in an m x n grid.

# Arguments
- `m::Int`: The number of rows in the grid.
- `n::Int`: The number of columns in the grid.

# Returns
A `LabelledGraph` representing a grid graph with nodes arranged in an m x n grid. 
Each node is labeled with its coordinates (m, n), where m is the row index and n is the column index.
"""
function SquareSingleNode(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


site(::Type{Dense}) = :site


site(::Type{Sparse}) = :sparse_site

"""
$(TYPEDSIGNATURES)

Assigns type of tensor to a PEPS node coordinates for a given Layout and Sparsity.
"""
function tensor_map(
    ::Type{SquareSingleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:Union{GaugesEnergy,EnergyGauges},S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols
            push!(map, PEPSNode(i, j + 1 // 2) => :central_h_single_node)
        end
        if i < nrows
            push!(map, PEPSNode(i + 1 // 2, j) => :central_v_single_node)
        end
    end
    map
end

"""
$(TYPEDSIGNATURES)

Assigns type of tensor to a PEPS node coordinates for a given Layout and Sparsity.
"""
function tensor_map(
    ::Type{SquareSingleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:EngGaugesEng,S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site(S))
        if j < ncols
            push!(map, PEPSNode(i, j + 1 // 2) => :central_h_single_node)
        end
        if i < nrows
            push!(
                map,
                PEPSNode(i + 1 // 5, j) => :sqrt_up,
                PEPSNode(i + 4 // 5, j) => :sqrt_down,
            )
        end
    end
    map
end

"""
$(TYPEDSIGNATURES)

Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(
    ::Type{SquareSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:GaugesEnergy}
    [
        GaugeInfo(
            (PEPSNode(i + 1 // 6, j), PEPSNode(i + 2 // 6, j)),
            PEPSNode(i + 1 // 2, j),
            1,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(
    ::Type{SquareSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:EnergyGauges}
    [
        GaugeInfo(
            (PEPSNode(i + 4 // 6, j), PEPSNode(i + 5 // 6, j)),
            PEPSNode(i + 1 // 2, j),
            2,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(
    ::Type{SquareSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:EngGaugesEng}
    [
        GaugeInfo(
            (PEPSNode(i + 2 // 5, j), PEPSNode(i + 3 // 5, j)),
            PEPSNode(i + 1 // 5, j),
            2,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareSingleNode geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareSingleNode{EnergyGauges}}
    main = Dict{Site,Sites}(i => (-1 // 6, 0, 3 // 6, 4 // 6) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(main, i + 1 // 2 => (0,))
    end

    right = Dict{Site,Sites}(i => (-3 // 6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(right, i + 1 // 2 => (0,))
    end

    MpoLayers(main, Dict(i => (3 // 6, 4 // 6) for i ∈ 1:ncols), right)
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareSingleNode geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareSingleNode{GaugesEnergy}}
    main = Dict{Site,Sites}(i => (-4 // 6, -1 // 2, 0, 1 // 6) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(main, i + 1 // 2 => (0,))
    end

    right = Dict{Site,Sites}(i => (-3 // 6, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(right, i + 1 // 2 => (0,))
    end

    MpoLayers(main, Dict(i => (1 // 6,) for i ∈ 1:ncols), right)
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareSingleNode geometry with the EngGaugesEng layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareSingleNode{EngGaugesEng}}
    main = Dict{Site,Sites}(i => (-2 // 5, -1 // 5, 0, 1 // 5, 2 // 5) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(main, i + 1 // 2 => (0,))
    end

    right = Dict{Site,Sites}(i => (-4 // 5, -1 // 5, 0) for i ∈ 1:ncols)
    for i ∈ 1:ncols-1
        push!(right, i + 1 // 2 => (0,))
    end

    MpoLayers(main, Dict(i => (1 // 5, 2 // 5) for i ∈ 1:ncols), right)
end

function projected_energy(
    net::PEPSNetwork,
    v::T,
    w::T,
    k::Int,
) where {T<:NTuple{N,Int} where {N}}
    en = interaction_energy(net, v, w)
    @inbounds en[projector(net, v, w), k]
end

function projected_energy(net::PEPSNetwork, v::T, w::T) where {T<:NTuple{N,Int} where {N}}
    en = interaction_energy(net, v, w)

    @inbounds en[projector(net, v, w), projector(net, w, v)]
end

"""
$(TYPEDSIGNATURES)

Calculates conditional probability for a SquareSingleNode Layout.
"""
function conditional_probability(
    ::Type{T},
    ctr::MpsContractor{S},
    ∂v::Vector{Int},
) where {T<:SquareSingleNode,S}
    β = ctr.beta
    i, j = ctr.current_node

    L = left_env(ctr, i, ∂v[1:j-1])
    R = right_env(ctr, i, ∂v[(j+2):ctr.peps.ncols+1])
    if ctr.onGPU
        R = CuArray(R)
    end
    M = dressed_mps(ctr, i)[j]

    @tensor LM[y, z] := L[x] * M[x, y, z]

    @nexprs 2 k ->
        (en_k = projected_energy(ctr.peps, (i, j), (i + 1 - k, j - 2 + k), ∂v[j-1+k]);
        p_k = projector(ctr.peps, (i, j), (i + 2 - k, j - 1 + k)))
    probs = probability(local_energy(ctr.peps, (i, j)) .+ en_1 .+ en_2, β)

    A = LM[:, p_1] .* R[:, p_2]
    # @reduce bnd_exp[x] := sum(y) A[y, x]
    bnd_exp = dropdims(sum(A; dims = 1); dims = 1)
    probs .*= Array(bnd_exp)

    push!(ctr.statistics, ((i, j), ∂v) => error_measure(probs))
    normalize_probability(probs)
end



function projectors_site_tensor(
    network::PEPSNetwork{T,S},
    vertex::Node,
) where {T<:SquareSingleNode,S}
    i, j = vertex
    projector.(Ref(network), Ref(vertex), ((i, j - 1), (i - 1, j), (i, j + 1), (i + 1, j)))
end


function nodes_search_order_Mps(peps::PEPSNetwork{T,S}) where {T<:SquareSingleNode,S}
    ([(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols], (peps.nrows + 1, 1))
end


function boundary(
    ::Type{T},
    ctr::MpsContractor{S},
    node::Node,
) where {T<:SquareSingleNode,S}
    i, j = node
    vcat(
        [((i, k), (i + 1, k)) for k ∈ 1:j-1]...,
        ((i, j - 1), (i, j)),
        [((i - 1, k), (i, k)) for k ∈ j:ctr.peps.ncols]...,
    )
end


function update_energy(
    ::Type{T},
    ctr::MpsContractor{S},
    σ::Vector{Int},
) where {T<:SquareSingleNode,S}
    net = ctr.peps
    i, j = ctr.current_node
    en = local_energy(net, (i, j))
    for v ∈ ((i, j - 1), (i - 1, j))
        en += bond_energy(net, (i, j), v, local_state_for_node(ctr, σ, v))
    end
    en
end
