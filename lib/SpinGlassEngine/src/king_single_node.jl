export KingSingleNode

"""
$(TYPEDSIGNATURES)

A geometric structure representing a 1-layer grid with nodes arranged in a grid of rows and columns,
and additional diagonal edges forming a cross pattern between neighboring nodes.

# Type Parameters
- `T <: AbstractTensorsLayout`: The layout of decomposition of tensors into MPS. Can be `GaugesEnergy`, `EnergyGauges` or `EngGaugesEng`.

# Constructors
- `KingSingleNode(layout::T)`: Create a `KingSingleNode` with the specified tensor layout.
"""
struct KingSingleNode{T<:AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)

Create a labeled grid graph with nodes arranged in an m x n grid and additional diagonal
edges forming a cross pattern between neighboring nodes.

# Arguments
- `m::Int`: The number of rows in the grid.
- `n::Int`: The number of columns in the grid.

# Returns
A `LabelledGraph` representing a grid graph with nodes arranged in an m x n grid,
and additional diagonal edges forming a cross pattern between neighboring nodes.
"""
function KingSingleNode(m::Int, n::Int)
    lg = SquareSingleNode(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j), (i + 1, j + 1))
        add_edge!(lg, (i + 1, j), (i, j + 1))
    end
    lg
end


VirtualSingleNode(::Type{Dense}) = :virtual_single_node


VirtualSingleNode(::Type{Sparse}) = :sparse_virtual_single_node


function tensor_map(
    ::Type{KingSingleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:Union{EnergyGauges,GaugesEnergy},S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1 // 2) => VirtualSingleNode(S),
            PEPSNode(i + 1 // 2, j) => :central_v_single_node,
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(map, PEPSNode(i + 1 // 2, j + 1 // 2) => :central_d_single_node)
    end
    map
end


function tensor_map(
    ::Type{KingSingleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:EngGaugesEng,S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site(S),
            PEPSNode(i, j - 1 // 2) => VirtualSingleNode(S),
            PEPSNode(i + 1 // 5, j) => :sqrt_up,
            PEPSNode(i + 4 // 5, j) => :sqrt_down,
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1
        push!(
            map,
            PEPSNode(i + 1 // 5, j + 1 // 2) => :sqrt_up_d,
            PEPSNode(i + 4 // 5, j + 1 // 2) => :sqrt_down_d,
        )
    end
    map
end


function gauges_list(
    ::Type{KingSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:GaugesEnergy}
    [
        GaugeInfo(
            (PEPSNode(i + 1 // 6, j), PEPSNode(i + 2 // 6, j)),
            PEPSNode(i + 1 // 2, j),
            1,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end


function gauges_list(
    ::Type{KingSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:EnergyGauges}
    [
        GaugeInfo(
            (PEPSNode(i + 4 // 6, j), PEPSNode(i + 5 // 6, j)),
            PEPSNode(i + 1 // 2, j),
            2,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end


function gauges_list(
    ::Type{KingSingleNode{T}},
    nrows::Int,
    ncols::Int,
) where {T<:EngGaugesEng}
    [
        GaugeInfo(
            (PEPSNode(i + 2 // 5, j), PEPSNode(i + 3 // 5, j)),
            PEPSNode(i + 1 // 5, j),
            2,
            :gauge_h,
        ) for i ∈ 1:nrows-1 for j ∈ 1//2:1//2:ncols
    ]
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the KingSingleNode geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:KingSingleNode{EnergyGauges}}
    MpoLayers(
        Dict(site(i) => (-1 // 6, 0, 3 // 6, 4 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (3 // 6, 4 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3 // 6, 0) for i ∈ 1//2:1//2:ncols),
    )
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the KingSingleNode geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:KingSingleNode{GaugesEnergy}}
    MpoLayers(
        Dict(site(i) => (-4 // 6, -1 // 2, 0, 1 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1 // 6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3 // 6, 0) for i ∈ 1//2:1//2:ncols),
    )
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the KingSingleNode geometry with the EngGaugesEng layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:KingSingleNode{EngGaugesEng}}
    MpoLayers(
        Dict(site(i) => (-2 // 5, -1 // 5, 0, 1 // 5, 2 // 5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1 // 5, 2 // 5) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-4 // 5, -1 // 5, 0) for i ∈ 1//2:1//2:ncols),
    )
end


# TODO: rewrite this using brodcasting if possible
function conditional_probability(
    ::Type{T},
    ctr::MpsContractor{S},
    ∂v::Vector{Int},
) where {T<:KingSingleNode,S}

    β = ctr.beta
    i, j = ctr.current_node

    L = left_env(ctr, i, ∂v[1:2*j-2])
    R = right_env(ctr, i, ∂v[(2*j+3):2*ctr.peps.ncols+2])

    ψ = dressed_mps(ctr, i)

    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    v = ((i, j - 1), (i - 1, j - 1), (i - 1, j))
    @nexprs 3 k -> (en_k = projected_energy(ctr.peps, (i, j), v[k], ∂v[2*j-1+k]))
    probs = probability(local_energy(ctr.peps, (i, j)) .+ en_1 .+ en_2 .+ en_3, β)

    p_rb = projector(ctr.peps, (i, j), (i + 1, j - 1))
    pr = projector(ctr.peps, (i, j), @ntuple 3 k -> (i + 2 - k, j + 1))
    pd = projector(ctr.peps, (i, j), (i + 1, j))

    # @cast lmx2[d, b, c] := LMX[d, (b, c)] (c ∈ 1:maximum(p_rb))
    c = maximum(p_rb)
    lmx2 = reshape(LMX, size(LMX, 1), size(LMX, 2) ÷ c, c)
    lmx2, M, R = Array.((lmx2, M, R))  # REWRITE

    for σ ∈ 1:length(probs)   # REWRITE on CUDA + parallelize
        lmx = @inbounds lmx2[:, ∂v[2*j-1], p_rb[σ]]
        m = @inbounds M[:, :, pd[σ]]
        r = @inbounds R[:, pr[σ]]
        @inbounds probs[σ] *= (lmx'*m*r)[]
    end

    push!(ctr.statistics, ((i, j), ∂v) => error_measure(probs))
    normalize_probability(probs)
end


function projectors_site_tensor(
    network::PEPSNetwork{T,S},
    vertex::Node,
) where {T<:KingSingleNode,S}
    i, j = vertex
    nbrs = (
        (@ntuple 3 k -> (i + 2 - k, j - 1)),
        (i - 1, j),
        (@ntuple 3 k -> (i + 2 - k, j + 1)),
        (i + 1, j),
    )
    projector.(Ref(network), Ref(vertex), nbrs)
end


function nodes_search_order_Mps(peps::PEPSNetwork{T,S}) where {T<:KingSingleNode,S}
    ([(i, j) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols], (peps.nrows + 1, 1))
end


function boundary(::Type{T}, ctr::MpsContractor{S}, node::Node) where {T<:KingSingleNode,S}
    i, j = node
    vcat(
        [
            [((i, k - 1), (i + 1, k), (i, k), (i + 1, k - 1)), ((i, k), (i + 1, k))] for
            k ∈ 1:(j-1)
        ]...,
        ((i, j - 1), (i + 1, j)),
        ((i, j - 1), (i, j)),
        ((i - 1, j - 1), (i, j)),
        ((i - 1, j), (i, j)),
        [
            [((i - 1, k - 1), (i, k), (i - 1, k), (i, k - 1)), ((i - 1, k), (i, k))] for
            k ∈ (j+1):ctr.peps.ncols
        ]...,
    )
end


function update_energy(
    ::Type{T},
    ctr::MpsContractor{S},
    σ::Vector{Int},
) where {T<:KingSingleNode,S}
    net = ctr.peps
    i, j = ctr.current_node
    en = local_energy(net, (i, j))
    for v ∈ ((i, j - 1), (i - 1, j), (i - 1, j - 1), (i - 1, j + 1))
        en += bond_energy(net, (i, j), v, local_state_for_node(ctr, σ, v))
    end
    en
end


function tensor(
    net::PEPSNetwork{KingSingleNode{T},S},
    node::PEPSNode,
    β::Real,
    ::Val{:central_d_single_node},
) where {T<:AbstractTensorsLayout,S<:AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    T_NW_SE = connecting_tensor(net, (i, j), (i + 1, j + 1), β)
    T_NE_SW = connecting_tensor(net, (i, j + 1), (i + 1, j), β)
    # @cast A[(u, uu), (dd, d)] := T_NW_SE[u, d] * T_NE_SW[uu, dd]
    u, d = size(T_NW_SE)
    uu, dd = size(T_NE_SW)
    A = reshape(reshape(T_NW_SE, u, 1, 1, d) .* reshape(T_NE_SW, 1, dd, uu), u * uu, d * dd)
    A
end


function Base.size(
    network::PEPSNetwork{KingSingleNode{T},S},
    node::PEPSNode,
    ::Val{:central_d_single_node},
) where {T<:AbstractTensorsLayout,S<:AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    s_NW_SE = size(interaction_energy(network, (i, j), (i + 1, j + 1)))
    s_NE_SW = size(interaction_energy(network, (i, j + 1), (i + 1, j)))
    (s_NW_SE[1] * s_NE_SW[1], s_NW_SE[2] * s_NE_SW[2])
end


function tensor(
    net::PEPSNetwork{KingSingleNode{T},S},
    node::PEPSNode,
    β::Real,
    ::Val{:sparse_virtual_single_node},
) where {T<:AbstractTensorsLayout,S<:Union{Sparse,Dense}}
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    pl = last(fuse_projectors(
        @ntuple 3 k -> projector(net, (i, j), (i + 2 - k, j + 1)) # p_lb, p_l, p_lt
    ))
    pr = last(fuse_projectors(
        @ntuple 3 k -> projector(net, (i, j + 1), (i + 2 - k, j)) # p_rb, p_r, p_rt
    ))
    VirtualTensor(
        net.lp,
        connecting_tensor(net, floor.(Int, v), ceil.(Int, v), β),
        (pl..., pr...),
    )
end


function tensor(
    net::PEPSNetwork{KingSingleNode{T},Dense},
    node::PEPSNode,
    β::Real,
    ::Val{:virtual_single_node},
) where {T<:AbstractTensorsLayout}
    sp = tensor(net, node, β, Val(:sparse_virtual_single_node))
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = (get_projector!(net.lp, x) for x in sp.projs)

    A = zeros(
        eltype(sp.con),
        length(p_l),
        maximum.((p_lt, p_rt))...,
        length(p_r),
        maximum.((p_lb, p_rb))...,
    )
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        @inbounds A[l, p_lt[l], p_rt[r], r, p_lb[l], p_rb[r]] = sp.con[p_l[l], p_r[r]]
    end
    # @cast B[l, (uu, u), r, (dd, d)] := A[l, uu, u, r, dd, d]
    B = reshape(A, size(A, 1), size(A, 2) * size(A, 3), size(A, 4), size(A, 5) * size(A, 6))
    B
end
