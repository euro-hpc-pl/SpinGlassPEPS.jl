export SquareDoubleNode, tensor, site_double_node

"""
$(TYPEDSIGNATURES)

A geometric structure representing a 2-layer grid with nodes arranged in rows and columns. 
Each node is labeled with a tuple (i, j, k), where i is the row index, j is the column index, and k is the layer index (1 or 2).

# Type Parameters
- `T <: AbstractTensorsLayout`: The layout of decomposition of tensors into MPS. Can be `GaugesEnergy`, `EnergyGauges` or `EngGaugesEng`.

# Constructors
- `SquareDoubleNode(layout::T)`: Create a `SquareDoubleNode` with the specified tensor layout.
"""
struct SquareDoubleNode{T<:AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)

Create a labeled graph representing a 2-layer grid graph with nodes arranged in an m x n grid. 
Each node is labeled with a tuple (i, j, k), where i is the row index, j is the column index, and k is the layer index (1 or 2).

# Arguments
- `m::Int`: The number of rows in the grid.
- `n::Int`: The number of columns in the grid.

# Returns
A `LabelledGraph` representing a 2-layer grid graph with nodes arranged in an m x n grid. 
Each node in layer 1 is connected to the corresponding node in layer 2, and neighboring nodes in both layers are connected.
"""
function SquareDoubleNode(m::Int, n::Int)
    labels = [(i, j, k) for j ∈ 1:n for i ∈ 1:m for k ∈ 1:2]
    lg = LabelledGraph(labels)
    for i ∈ 1:m, j ∈ 1:n
        add_edge!(lg, (i, j, 1), (i, j, 2))
    end

    for i ∈ 1:m-1, j ∈ 1:n
        add_edge!(lg, (i, j, 1), (i + 1, j, 1))
        add_edge!(lg, (i, j, 1), (i + 1, j, 2))
        add_edge!(lg, (i, j, 2), (i + 1, j, 1))
        add_edge!(lg, (i, j, 2), (i + 1, j, 2))
    end

    for i ∈ 1:m, j ∈ 1:n-1
        add_edge!(lg, (i, j, 2), (i, j + 1, 2))
        add_edge!(lg, (i, j, 2), (i, j + 1, 1))
        add_edge!(lg, (i, j, 1), (i, j + 1, 2))
        add_edge!(lg, (i, j, 1), (i, j + 1, 1))
    end
    lg
end


site_double_node(::Type{Dense}) = :site_square_double_node


site_double_node(::Type{Sparse}) = :sparse_site_square_double_node


function tensor_map(
    ::Type{SquareDoubleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:Union{GaugesEnergy,EnergyGauges},S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()
    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(map, PEPSNode(i, j) => site_double_node(S))
        if j < ncols
            push!(map, PEPSNode(i, j + 1 // 2) => :central_h_double_node)
        end
        if i < nrows
            push!(map, PEPSNode(i + 1 // 2, j) => :central_v_double_node)
        end
    end
    map
end


"""
$(TYPEDSIGNATURES)
Assigns gauges and corresponding information to GaugeInfo structure for a given Layout.
"""
function gauges_list(
    ::Type{SquareDoubleNode{T}},
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
    ::Type{SquareDoubleNode{T}},
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
Defines the MPO layers for the SquareSingleNode geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareDoubleNode{EnergyGauges}}
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
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareDoubleNode{GaugesEnergy}}
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


function conditional_probability(
    ::Type{T},
    ctr::MpsContractor{S},
    ∂v::Vector{Int},
) where {T<:SquareDoubleNode,S}
    β = ctr.beta
    i, j, k = ctr.current_node

    L = left_env(ctr, i, ∂v[1:j-1])
    M = dressed_mps(ctr, i)[j]
    @tensor LM[y, z] := L[x] * M[x, y, z]

    if k == 1  # here has to avarage over s2
        R = right_env(ctr, i, ∂v[(j+8):end])
        if ctr.onGPU
            R = CuArray(R)
        end

        eng_loc = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]
        el = [interaction_energy(ctr.peps, (i, j, k), (i, j - 1, m)) for k ∈ 1:2, m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, k), (i, j - 1, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_l = [@view el[k, m][pl[k, m], ∂v[j-1+k+(m-1)*2]] for k ∈ 1:2, m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, k), (i - 1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, k), (i - 1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_u = [@view eu[k, m][pu[k, m], ∂v[j+3+k+(m-1)*2]] for k ∈ 1:2, m ∈ 1:2]

        en = [
            eng_loc[k] .+ eng_l[k, 1] .+ eng_l[k, 2] .+ eng_u[k, 1] .+ eng_u[k, 2] for
            k ∈ 1:2
        ]

        en12 = interaction_energy(ctr.peps, (i, j, 1), (i, j, 2))
        p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))

        pr1 = projector(ctr.peps, (i, j, 1), ((i, j + 1, 1), (i, j + 1, 2)))
        pr2 = projector(ctr.peps, (i, j, 2), ((i, j + 1, 1), (i, j + 1, 2)))
        pd1 = projector(ctr.peps, (i, j, 1), ((i + 1, j, 1), (i + 1, j, 2)))
        pd2 = projector(ctr.peps, (i, j, 2), ((i + 1, j, 1), (i + 1, j, 2)))

        le = reshape(en[1], (:, 1)) .+ en12[p12, p21] .+ reshape(en[2], (1, :))
        ele = exp.(-β .* (le .- minimum(le)))

        # @cast LM3[x, s1, s2] := LM[x, (s1, s2)] (s2 ∈ 1:maximum(pd2))
        s2 = maximum(pd2)
        LM3 = reshape(LM, size(LM, 1), size(LM, 2) ÷ s2, s2)
        # @cast R3[x, s1, s2] := R[x, (s1, s2)] (s2 ∈ 1:maximum(pr2))
        s2 = maximum(pr2)
        R3 = reshape(R, size(R, 1), size(R, 2) ÷ s2, s2)
        LR = dropdims(sum(LM3[:, pd1, pd2] .* R3[:, pr1, pr2], dims = 1), dims = 1)

        probs = dropdims(sum(Array(LR) .* ele, dims = 2), dims = 2)
    else  # k == 2 ; here s1 is fixed
        R = right_env(ctr, i, ∂v[(j+7):end])
        if ctr.onGPU
            R = CuArray(R)
        end
        eng_loc = local_energy(ctr.peps, (i, j, 2))

        el = [interaction_energy(ctr.peps, (i, j, 2), (i, j - 1, m)) for m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, 2), (i, j - 1, m)) for m ∈ 1:2]
        eng_l = [@view el[m][pl[m], ∂v[j-1+m]] for m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, 2), (i - 1, j, m)) for m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, 2), (i - 1, j, m)) for m ∈ 1:2]
        eng_u = [@view eu[m][pu[m], ∂v[j+1+m]] for m ∈ 1:2]

        e21 = interaction_energy(ctr.peps, (i, j, 2), (i, j, 1))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
        eng_21 = @inbounds @view e21[p21, ∂v[j+4]]

        le = eng_loc .+ eng_l[1] .+ eng_l[2] .+ eng_u[1] .+ eng_u[2] .+ eng_21
        ele = exp.(-β .* (le .- minimum(le)))

        pr2 = projector(ctr.peps, (i, j, 2), ((i, j + 1, 1), (i, j + 1, 2)))
        pd2 = projector(ctr.peps, (i, j, 2), ((i + 1, j, 1), (i + 1, j, 2)))

        # @cast R3[x, p1, p2] := R[x, (p1, p2)] (p2 ∈ 1:maximum(pr2))
        p2 = maximum(pr2)
        R3 = reshape(R, size(R, 1), size(R, 2) ÷ p2, p2)
        R2 = R3[:, ∂v[j+5], :]
        # @cast LM3[x, p1, p2] := LM[x, (p1, p2)] (p2 ∈ 1:maximum(pd2))
        p2 = maximum(pd2)
        LM3 = reshape(LM, size(LM, 1), size(LM, 2) ÷ p2, p2)
        LM2 = LM3[:, ∂v[j+6], :]
        LR = dropdims(sum(R2[:, pr2] .* LM2[:, pd2], dims = 1), dims = 1)
        probs = ele .* Array(LR)
    end

    push!(ctr.statistics, ((i, j, k), ∂v) => error_measure(probs))
    normalize_probability(probs)
end



function nodes_search_order_Mps(peps::PEPSNetwork{T,S}) where {T<:SquareDoubleNode,S}
    (
        [(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2],
        (peps.nrows + 1, 1, 1),
    )
end


function boundary(
    ::Type{T},
    ctr::MpsContractor{S},
    node::Node,
) where {T<:SquareDoubleNode,S}  # TODO
    i, j, k = node
    if k == 1
        bnd = vcat(
            [
                (
                    (i, m, 1),
                    ((i + 1, m, 1), (i + 1, m, 2)),
                    (i, m, 2),
                    ((i + 1, m, 1), (i + 1, m, 2)),
                ) for m ∈ 1:j-1
            ]...,
            ((i, j - 1, 1), (i, j, 1)),
            ((i, j - 1, 1), (i, j, 2)),
            ((i, j - 1, 2), (i, j, 1)),
            ((i, j - 1, 2), (i, j, 2)),
            ((i - 1, j, 1), (i, j, 1)),
            ((i - 1, j, 1), (i, j, 2)),
            ((i - 1, j, 2), (i, j, 1)),
            ((i - 1, j, 2), (i, j, 2)),
            [
                (
                    (i - 1, m, 1),
                    ((i, m, 1), (i, m, 2)),
                    (i - 1, m, 2),
                    ((i, m, 1), (i, m, 2)),
                ) for m ∈ j+1:ctr.peps.ncols
            ]...,
        )
    else  # k == 2
        bnd = vcat(
            [
                (
                    (i, m, 1),
                    ((i + 1, m, 1), (i + 1, m, 2)),
                    (i, m, 2),
                    ((i + 1, m, 1), (i + 1, m, 2)),
                ) for m ∈ 1:j-1
            ]...,
            ((i, j - 1, 1), (i, j, 2)),
            ((i, j - 1, 2), (i, j, 2)),
            ((i - 1, j, 1), (i, j, 2)),
            ((i - 1, j, 2), (i, j, 2)),
            ((i, j, 1), (i, j, 2)),
            ((i, j, 1), ((i, j + 1, 1), (i, j + 1, 2))),
            ((i, j, 1), ((i + 1, j, 1), (i + 1, j, 2))),
            [
                (
                    (i - 1, m, 1),
                    ((i, m, 1), (i, m, 2)),
                    (i - 1, m, 2),
                    ((i, m, 1), (i, m, 2)),
                ) for m ∈ j+1:ctr.peps.ncols
            ]...,
        )
    end
    bnd
end


function update_energy(
    ::Type{T},
    ctr::MpsContractor{S},
    σ::Vector{Int},
) where {T<:SquareDoubleNode,S}
    net = ctr.peps
    i, j, k = ctr.current_node

    en = local_energy(net, (i, j, k))
    for v ∈ ((i, j - 1, 1), (i, j - 1, 2), (i - 1, j, 1), (i - 1, j, 2))
        en += bond_energy(net, (i, j, k), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2
        return en
    end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))  # here k=2
    en
end


function tensor(
    net::PEPSNetwork{T,S},
    node::PEPSNode,
    β::Real,
    ::Val{:sparse_site_square_double_node},
) where {T<:AbstractGeometry,S}
    @nexprs 2 k -> (v_k = (node.i, node.j, k);
    en_k = local_energy(net, v_k))
    en12 = projected_energy(net, v_1, v_2)
    # @cast eloc12[(x, y)] := en12[x, y] + en_1[x] + en_2[y]
    eloc12 = vec((+).(en12, en_1, en_2'))

    SiteTensor(
        net.lp,
        exp.(-β .* (eloc12 .- minimum(eloc12))),
        projectors_site_tensor(net, Node(node)),
    )
end


function tensor(
    net::PEPSNetwork{T,S},
    node::PEPSNode,
    β::Real,
    ::Val{:site_square_double_node},
) where {T<:AbstractGeometry,S}
    sp = tensor(net, node, β, Val(:sparse_site_square_double_node))
    projs = Tuple(get_projector!(net.lp, x) for x in sp.projs)
    A = zeros(maximum.(projs))
    for (σ, lexp) ∈ enumerate(sp.loc_exp)
        @inbounds A[getindex.(projs, Ref(σ))...] += lexp
    end
    A
end


function tensor(
    network::PEPSNetwork{T,Sparse},
    node::PEPSNode,
    β::Real,
    ::Val{:central_h_double_node},
) where {T<:AbstractGeometry}
    i, j = node.i, floor(Int, node.j)
    CentralTensor(network, β, (i, j), (i, j + 1))
end


function tensor(
    network::PEPSNetwork{T,Dense},
    node::PEPSNode,
    β::Real,
    ::Val{:central_h_double_node},
) where {T<:AbstractGeometry}
    i, j = node.i, floor(Int, node.j)
    dense_central(CentralTensor(network, β, (i, j), (i, j + 1)))
end


function tensor(
    network::PEPSNetwork{T,Sparse},
    node::PEPSNode,
    β::Real,
    ::Val{:central_v_double_node},
) where {T<:AbstractGeometry}
    i, j = floor(Int, node.i), node.j
    CentralTensor(network, β, (i, j), (i + 1, j))
end


function tensor(
    network::PEPSNetwork{T,Dense},
    node::PEPSNode,
    β::Real,
    ::Val{:central_v_double_node},
) where {T<:AbstractGeometry}
    i, j = floor(Int, node.i), node.j
    dense_central(CentralTensor(network, β, (i, j), (i + 1, j)))
end


function CentralTensor(
    network::PEPSNetwork{T,S},
    β::Real,
    node1::NTuple{2,Int64},
    node2::NTuple{2,Int64},
) where {T<:AbstractGeometry,S}
    i1, j1 = node1
    i2, j2 = node2

    p21l = projector(network, (i1, j1, 2), (i2, j2, 1))
    p22l = projector(network, (i1, j1, 2), (i2, j2, 2))
    p12l = projector(network, (i1, j1, 1), (i2, j2, 2))
    p11l = projector(network, (i1, j1, 1), (i2, j2, 1))

    p1l, (p11l, p12l) = fuse_projectors((p11l, p12l))
    p2l, (p21l, p22l) = fuse_projectors((p21l, p22l))

    p11r = projector(network, (i2, j2, 1), (i1, j1, 1))
    p21r = projector(network, (i2, j2, 1), (i1, j1, 2))
    p12r = projector(network, (i2, j2, 2), (i1, j1, 1))
    p22r = projector(network, (i2, j2, 2), (i1, j1, 2))

    p1r, (p11r, p21r) = fuse_projectors((p11r, p21r))
    p2r, (p12r, p22r) = fuse_projectors((p12r, p22r))

    e11 = interaction_energy(network, (i1, j1, 1), (i2, j2, 1))
    e12 = interaction_energy(network, (i1, j1, 1), (i2, j2, 2))
    e21 = interaction_energy(network, (i1, j1, 2), (i2, j2, 1))
    e22 = interaction_energy(network, (i1, j1, 2), (i2, j2, 2))

    e11 = e11[p11l, p11r]
    e21 = e21[p21l, p21r]
    e12 = e12[p12l, p12r]
    e22 = e22[p22l, p22r]

    le11 = exp.(-β .* (e11 .- minimum(e11)))
    le21 = exp.(-β .* (e21 .- minimum(e21)))
    le12 = exp.(-β .* (e12 .- minimum(e12)))
    le22 = exp.(-β .* (e22 .- minimum(e22)))

    SpinGlassTensors.CentralTensor(le11, le12, le21, le22)
end

function CentralTensor_size(
    network::PEPSNetwork{T,S},
    node1::NTuple{2,Int64},
    node2::NTuple{2,Int64},
) where {T<:AbstractGeometry,S}
    i1, j1 = node1
    i2, j2 = node2

    p21l = projector(network, (i1, j1, 2), (i2, j2, 1))
    p22l = projector(network, (i1, j1, 2), (i2, j2, 2))
    p12l = projector(network, (i1, j1, 1), (i2, j2, 2))
    p11l = projector(network, (i1, j1, 1), (i2, j2, 1))

    p1l, _ = fuse_projectors((p11l, p12l))
    p2l, _ = fuse_projectors((p21l, p22l))

    p11r = projector(network, (i2, j2, 1), (i1, j1, 1))
    p21r = projector(network, (i2, j2, 1), (i1, j1, 2))
    p12r = projector(network, (i2, j2, 2), (i1, j1, 1))
    p22r = projector(network, (i2, j2, 2), (i1, j1, 2))

    p1r, _ = fuse_projectors((p11r, p21r))
    p2r, _ = fuse_projectors((p12r, p22r))

    sl = maximum(p1l) * maximum(p2l)
    sr = maximum(p1r) * maximum(p2r)
    (sl, sr)
end

function Base.size(
    network::AbstractGibbsNetwork{Node,PEPSNode},
    node::PEPSNode,
    ::Val{:central_v_double_node},
)
    i, j = floor(Int, node.i), node.j
    CentralTensor_size(network, (i, j), (i + 1, j))
end

function Base.size(
    network::AbstractGibbsNetwork{Node,PEPSNode},
    node::PEPSNode,
    ::Val{:central_h_double_node},
)
    i, j = node.i, floor(Int, node.j)
    CentralTensor_size(network, (i, j), (i, j + 1))
end



function projectors_site_tensor(
    net::PEPSNetwork{T,S},
    vertex::Node,
) where {T<:SquareDoubleNode,S}
    i, j = vertex
    (
        outer_projector(
            (projector(net, (i, j, k), ((i, j - 1, 1), (i, j - 1, 2))) for k ∈ 1:2)..., # l
        ),
        outer_projector(
            (projector(net, (i, j, k), ((i - 1, j, 1), (i - 1, j, 2))) for k ∈ 1:2)..., # t
        ),
        outer_projector(
            (projector(net, (i, j, k), ((i, j + 1, 1), (i, j + 1, 2))) for k ∈ 1:2)..., # r
        ),
        outer_projector(
            (projector(net, (i, j, k), ((i + 1, j, 1), (i + 1, j, 2))) for k ∈ 1:2)..., # b
        ),
    )
end
