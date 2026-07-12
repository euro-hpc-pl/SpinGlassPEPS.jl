export SquareCrossDoubleNode, precompute_conditional, VirtualDoubleNode


"""
$(TYPEDSIGNATURES)

A geometric structure representing a 2-layer grid with nodes arranged in rows and columns,
and additional diagonal edges forming a cross pattern between neighboring nodes.
Each node is labeled with a tuple (i, j, k), where i is the row index, j is the column index, and k is the layer index (1 or 2).

## Type Parameters
- `T <: AbstractTensorsLayout`: The layout of decomposition of tensors into MPS. Can be `GaugesEnergy`, `EnergyGauges` or `EngGaugesEng`.

# Constructors
- `KingSingleNode(layout::T)`: Create a `SquareCrossDoubleNode` with the specified tensor layout.

## Description
`SquareCrossDoubleNode` is a geometry type that models a double unit cell square lattice with diagonal interaction.
This geometry is suitable for systems with tensors laid out according to the specified `AbstractTensorsLayout`.
It can be used in Pegasus and Zephyr graphs.
"""
struct SquareCrossDoubleNode{T<:AbstractTensorsLayout} <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
Create a graph representing a double unit cell square lattice with diagonal interactions.

## Arguments
- `m::Int`: The number of rows in the lattice.
- `n::Int`: The number of columns in the lattice.

## Returns
- `AbstractGraph`: A graph representing the double unit cell square lattice with diagonal interactions.

## Description
The `SquareCrossDoubleNode` function creates a graph representing a double unit cell square lattice
where each lattice site interacts with its nearest neighbor and next nearest neighbor.
The lattice is specified by the number of rows (`m`) and columns (`n`).
The resulting graph is suitable for systems with tensors laid out according to the specified geometry.
"""
function SquareCrossDoubleNode(m::Int, n::Int)
    lg = SquareDoubleNode(m, n)
    for i ∈ 1:m-1, j ∈ 1:n-1
        add_edge!(lg, (i, j, 1), (i + 1, j + 1, 1))
        add_edge!(lg, (i, j, 1), (i + 1, j + 1, 2))
        add_edge!(lg, (i, j, 2), (i + 1, j + 1, 1))
        add_edge!(lg, (i, j, 2), (i + 1, j + 1, 2))
        add_edge!(lg, (i + 1, j, 1), (i, j + 1, 1))
        add_edge!(lg, (i + 1, j, 1), (i, j + 1, 2))
        add_edge!(lg, (i + 1, j, 2), (i, j + 1, 1))
        add_edge!(lg, (i + 1, j, 2), (i, j + 1, 2))
    end
    lg
end

"""
$(TYPEDSIGNATURES)
Create a symbol representing a virtual double node for a dense tensor layout.

## Arguments
- `::Type{Dense}`: The `Dense` tensor layout type.

## Returns
- `Symbol`: A symbol representing the virtual double node.

## Description
The `VirtualDoubleNode` function generates a symbol (`:virtual_double_node`) that represents a virtual double node in the context of a dense tensor layout.
This symbol is often used to indicate the presence of a virtual double node when working with certain tensors.
"""
VirtualDoubleNode(::Type{Dense}) = :virtual_double_node

"""
$(TYPEDSIGNATURES)
Create a symbol representing a virtual double node for a sparse tensor layout.

## Arguments
- `::Type{Sparse}`: The `Sparse` tensor layout type.

## Returns
- `Symbol`: A symbol representing the virtual double node.

## Description
The `VirtualDoubleNode` function generates a symbol (`:virtual_double_node`) that represents a virtual double node in the context of a sparse tensor layout.
This symbol is often used to indicate the presence of a virtual double node when working with certain tensors.
"""
VirtualDoubleNode(::Type{Sparse}) = :sparse_virtual_double_node

"""
$(TYPEDSIGNATURES)

Create a mapping of tensor network nodes for a square cross double node geometry.

## Arguments
- `::Type{SquareCrossDoubleNode{T}}`: Type representing a square cross double node geometry.
- `::Type{S}`: Type representing sparsity in the tensor network.
- `nrows::Int`: Number of rows in the tensor network.
- `ncols::Int`: Number of columns in the tensor network.

## Returns
- `Dict{PEPSNode, Symbol}`: A dictionary mapping PEPS nodes to symbols representing their corresponding tensor network nodes.

## Description
The `tensor_map` function generates a mapping of tensor network nodes for a square cross double node geometry.
The mapping includes different types of nodes, such as site double nodes, virtual double nodes, central vertical double nodes, and central diagonal double nodes.
"""
function tensor_map(
    ::Type{SquareCrossDoubleNode{T}},
    ::Type{S},
    nrows::Int,
    ncols::Int,
) where {T<:Union{EnergyGauges,GaugesEnergy},S<:AbstractSparsity}
    map = Dict{PEPSNode,Symbol}()

    for i ∈ 1:nrows, j ∈ 1:ncols
        push!(
            map,
            PEPSNode(i, j) => site_double_node(S),
            PEPSNode(i, j - 1 // 2) => VirtualDoubleNode(S),
            PEPSNode(i + 1 // 2, j) => :central_v_double_node,
        )
    end

    for i ∈ 1:nrows-1, j ∈ 0:ncols-1   # why from 0?
        push!(map, PEPSNode(i + 1 // 2, j + 1 // 2) => :central_d_double_node)
    end
    map
end


"""
$(TYPEDSIGNATURES)
Create a list of gauge information for a square cross double node geometry and GaugesEnergy Layout.

## Arguments
- `::Type{SquareCrossDoubleNode{T}}`: Type representing a square cross double node geometry.
- `nrows::Int`: Number of rows in the tensor network.
- `ncols::Int`: Number of columns in the tensor network.

## Returns
- `Vector{GaugeInfo}`: A vector of `GaugeInfo` objects representing gauge information for the specified geometry.

## Description
The `gauges_list` function generates a list of `GaugeInfo` objects for a square cross double node geometry.
Each `GaugeInfo` object contains information about the positions of gauge links, the position of the attached tensor, the leg index, and the type of gauge.
"""
function gauges_list(
    ::Type{SquareCrossDoubleNode{T}},
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

"""
$(TYPEDSIGNATURES)
Create a list of gauge information for a square cross double node geometry and EnergyGauges Layout.

## Arguments
- `::Type{SquareCrossDoubleNode{T}}`: Type representing a square cross double node geometry.
- `nrows::Int`: Number of rows in the tensor network.
- `ncols::Int`: Number of columns in the tensor network.

## Returns
- `Vector{GaugeInfo}`: A vector of `GaugeInfo` objects representing gauge information for the specified geometry.

## Description
The `gauges_list` function generates a list of `GaugeInfo` objects for a square cross double node geometry.
Each `GaugeInfo` object contains information about the positions of gauge links, the position of the attached tensor, the leg index, and the type of gauge.
"""
function gauges_list(
    ::Type{SquareCrossDoubleNode{T}},
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


"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareCrossDoubleNode geometry with the EnergyGauges layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareCrossDoubleNode{EnergyGauges}}
    MpoLayers(
        Dict(site(i) => (-1 // 6, 0, 3 // 6, 4 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (3 // 6, 4 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3 // 6, 0) for i ∈ 1//2:1//2:ncols),
    )
end

"""
$(TYPEDSIGNATURES)

Defines the MPO layers for the SquareCrossDoubleNode geometry with the GaugesEnergy layout.
"""
function MpoLayers(::Type{T}, ncols::Int) where {T<:SquareCrossDoubleNode{GaugesEnergy}}
    MpoLayers(
        Dict(site(i) => (-4 // 6, -1 // 2, 0, 1 // 6) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (1 // 6,) for i ∈ 1//2:1//2:ncols),
        Dict(site(i) => (-3 // 6, 0) for i ∈ 1//2:1//2:ncols),
    )
end

"""
$(TYPEDSIGNATURES)

Precompute conditional probabilities and energies for a square cross double node tensor contraction.

## Arguments
- `::Type{T}`: Type representing a square cross double node tensor network.
- `ctr::MpsContractor{S}`: Tensor contractor for the tensor network.
- `current_node`: Current node position in the tensor network.

## Returns
- Tuple: A tuple containing precomputed conditional probabilities and energies.

## Description
The `precompute_conditional` function computes and returns precomputed conditional probabilities and energies for the specified square cross double node tensor contraction.
It takes into account the geometry of the tensor network, interaction energies, and projectors.
The precomputed values are used during the tensor contraction process to speed up the computation.
The function is specialized for the `SquareCrossDoubleNode` tensor network type and is parametrized by the layout type `S` of the contractor.
"""
@memoize Dict function precompute_conditional(
    ::Type{T},
    ctr::MpsContractor{S},
    current_node,
) where {T<:SquareCrossDoubleNode,S}
    i, j, k = current_node
    β = ctr.beta
    if k == 1
        en12 = interaction_energy(ctr.peps, (i, j, 1), (i, j, 2))
        p12 = projector(ctr.peps, (i, j, 1), (i, j, 2))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))

        plb1 = projector(ctr.peps, (i, j, 1), ((i + 1, j - 1, 1), (i + 1, j - 1, 2)))
        plb2 = projector(ctr.peps, (i, j, 2), ((i + 1, j - 1, 1), (i + 1, j - 1, 2)))
        prf1 = projector(
            ctr.peps,
            (i, j, 1),
            (
                (i + 1, j + 1, 1),
                (i + 1, j + 1, 2),
                (i, j + 1, 1),
                (i, j + 1, 2),
                (i - 1, j + 1, 1),
                (i - 1, j + 1, 2),
            ),
        )
        prf2 = projector(
            ctr.peps,
            (i, j, 2),
            (
                (i + 1, j + 1, 1),
                (i + 1, j + 1, 2),
                (i, j + 1, 1),
                (i, j + 1, 2),
                (i - 1, j + 1, 1),
                (i - 1, j + 1, 2),
            ),
        )
        pd1 = projector(ctr.peps, (i, j, 1), ((i + 1, j, 1), (i + 1, j, 2)))
        pd2 = projector(ctr.peps, (i, j, 2), ((i + 1, j, 1), (i + 1, j, 2)))

        plb = outer_projector(plb1, plb2)
        prf = outer_projector(prf1, prf2)
        pd = outer_projector(pd1, pd2)

        eng_loc = [local_energy(ctr.peps, (i, j, k)) for k ∈ 1:2]

        el = [interaction_energy(ctr.peps, (i, j, k), (i, j - 1, m)) for k ∈ 1:2, m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, k), (i, j - 1, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_l = [el[k, m][pl[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        elu = [
            interaction_energy(ctr.peps, (i, j, k), (i - 1, j - 1, m)) for k ∈ 1:2, m ∈ 1:2
        ]
        plu = [projector(ctr.peps, (i, j, k), (i - 1, j - 1, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_lu = [elu[k, m][plu[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, k), (i - 1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, k), (i - 1, j, m)) for k ∈ 1:2, m ∈ 1:2]
        eng_u = [eu[k, m][pu[k, m], :] for k ∈ 1:2, m ∈ 1:2]

        le = en12[p12, p21]
        ele = exp.(-β .* (le .- minimum(le)))

        splb = maximum(plb)

        if ctr.onGPU
            ele = CuArray(ele)
            eng_loc = [CuArray(eng_loc[k]) for k ∈ 1:2]
            eng_l = [CuArray(eng_l[k, m]) for k ∈ 1:2, m ∈ 1:2]
            eng_lu = [CuArray(eng_lu[k, m]) for k ∈ 1:2, m ∈ 1:2]
            eng_u = [CuArray(eng_u[k, m]) for k ∈ 1:2, m ∈ 1:2]
            plb = CuArray(plb)
            prf = CuArray(prf)
            pd = CuArray(pd)
        end

        return (ele, eng_loc, eng_l, eng_lu, eng_u, plb, prf, pd, splb)
    else
        eng_loc = local_energy(ctr.peps, (i, j, 2))

        el = [interaction_energy(ctr.peps, (i, j, 2), (i, j - 1, m)) for m ∈ 1:2]
        pl = [projector(ctr.peps, (i, j, 2), (i, j - 1, m)) for m ∈ 1:2]
        eng_l = [el[m][pl[m], :] for m ∈ 1:2]

        elu = [interaction_energy(ctr.peps, (i, j, 2), (i - 1, j - 1, m)) for m ∈ 1:2]
        plu = [projector(ctr.peps, (i, j, 2), (i - 1, j - 1, m)) for m ∈ 1:2]
        eng_lu = [elu[m][plu[m], :] for m ∈ 1:2]

        eu = [interaction_energy(ctr.peps, (i, j, 2), (i - 1, j, m)) for m ∈ 1:2]
        pu = [projector(ctr.peps, (i, j, 2), (i - 1, j, m)) for m ∈ 1:2]
        eng_u = [eu[m][pu[m], :] for m ∈ 1:2]

        en12 = interaction_energy(ctr.peps, (i, j, 1), (i, j, 2))
        p21 = projector(ctr.peps, (i, j, 2), (i, j, 1))
        eng_12 = @view en12[:, p21]

        plb1 = projector(ctr.peps, (i, j, 1), ((i + 1, j - 1, 1), (i + 1, j - 1, 2)))
        plb2 = projector(ctr.peps, (i, j, 2), ((i + 1, j - 1, 1), (i + 1, j - 1, 2)))
        prf2 = projector(
            ctr.peps,
            (i, j, 2),
            (
                (i + 1, j + 1, 1),
                (i + 1, j + 1, 2),
                (i, j + 1, 1),
                (i, j + 1, 2),
                (i - 1, j + 1, 1),
                (i - 1, j + 1, 2),
            ),
        )
        pd2 = projector(ctr.peps, (i, j, 2), ((i + 1, j, 1), (i + 1, j, 2)))

        splb1, splb2, sprf2, spd2 = maximum.((plb1, plb2, prf2, pd2))

        if ctr.onGPU
            eng_loc = CuArray(eng_loc)
            eng_l = [CuArray(eng_l[m]) for m ∈ 1:2]
            eng_lu = [CuArray(eng_lu[m]) for m ∈ 1:2]
            eng_u = [CuArray(eng_u[m]) for m ∈ 1:2]
            eng_12 = CuArray(eng_12)
            plb2 = CuArray(plb2)
            prf2 = CuArray(prf2)
            pd2 = CuArray(pd2)
        end
        return (
            eng_loc,
            eng_l,
            eng_lu,
            eng_u,
            eng_12,
            plb2,
            prf2,
            pd2,
            splb1,
            splb2,
            sprf2,
            spd2,
        )
    end
end

"""
$(TYPEDSIGNATURES)
Compute the conditional probability of states for a square cross double node tensor geometry.

## Arguments
- `::Type{T}`: Type representing a square cross double node tensor network.
- `ctr::MpsContractor{S}`: Tensor contractor for the tensor network.
- `∂v::Vector{Int}`: Vector of indices representing the contracted environment indices.

## Returns
- Vector{Float64}: Conditional probabilities for different states.

## Description
The `conditional_probability` function computes the conditional probabilities of different states for a specified square cross double node tensor geometry.
It takes into account the geometry of the tensor network, interaction energies, and precomputed values.
The function supports both left and right environments, and the resulting probabilities are normalized.
The function is specialized for the `SquareCrossDoubleNode` tensor network type and is parametrized by the layout type `S` of the contractor.
"""
function conditional_probability(
    ::Type{T},
    ctr::MpsContractor{S},
    ∂v::Vector{Int},
) where {T<:SquareCrossDoubleNode,S}
    β = ctr.beta
    i, j, k = ctr.current_node

    L = left_env(ctr, i, ∂v[1:2*j-2])
    ψ = dressed_mps(ctr, i)
    MX, M = ψ[j-1//2], ψ[j]
    @tensor LMX[y, z] := L[x] * MX[x, y, z]

    if k == 1  # here has to avarage over s2
        R = right_env(ctr, i, ∂v[(2*j+12):end])
        if ctr.onGPU
            R = CuArray(R)
        end

        ele, eng_loc, eng_l, eng_lu, eng_u, plb, prf, pd, splb =
            precompute_conditional(T, ctr, ctr.current_node)

        eng_l = [@view eng_l[k, m][:, ∂v[2*j-1+k+(m-1)*2]] for k ∈ 1:2, m ∈ 1:2]
        eng_lu = [@view eng_lu[k, m][:, ∂v[2*j+3+k+(m-1)*2]] for k ∈ 1:2, m ∈ 1:2]
        eng_u = [@view eng_u[k, m][:, ∂v[2*j+7+k+(m-1)*2]] for k ∈ 1:2, m ∈ 1:2]
        en = [
            eng_loc[k] .+ eng_l[k, 1] .+ eng_l[k, 2] .+ eng_lu[k, 1] .+ eng_lu[k, 2] .+
            eng_u[k, 1] .+ eng_u[k, 2] for k ∈ 1:2
        ]
        en = [exp.(-β .* (en[k] .- minimum(en[k]))) for k ∈ 1:2]
        tele = reshape(en[1], (:, 1)) .* ele .* reshape(en[2], (1, :))

        # @cast LMX4[x, y, v, p] := LMX[(x, y), (v, p)] (x ∈ 1:1, p ∈ 1:splb)
        LMX4 = reshape(LMX, 1, size(LMX, 1), size(LMX, 2) ÷ splb, splb)
        LMX3 = LMX4[:, :, ∂v[2*j-1], :]
        # @cast R3[x, y, p] := R[(x, y), p] (y ∈ 1:1)
        R3 = reshape(R, size(R, 1), 1, size(R, 2))

        from = 1
        total_size = length(pd)

        total_memory = 2^33
        s1, s2, _ = size(M)
        batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s1 + s2 + 1)))), 1)
        batch_size = Int(2^floor(log2(batch_size) + 1e-6))
        LR =
            typeof(M) <: CuArray ? CUDA.zeros(eltype(M), total_size) :
            zeros(eltype(M), total_size)
        while from <= total_size
            to = min(total_size, from + batch_size - 1)
            vplb = @view plb[from:to]
            vpd = @view pd[from:to]
            vprf = @view prf[from:to]
            @inbounds LMX3p = LMX3[:, :, vplb]
            @inbounds Mp = M[:, :, vpd]
            @inbounds R3p = R3[:, :, vprf]
            @inbounds LR[from:to] = dropdims(LMX3p ⊠ Mp ⊠ R3p, dims = (1, 2))
            from = to + 1
        end
        LR = reshape(LR, size(tele))
        LR .*= tele
        probs = Array(dropdims(sum(LR, dims = 2), dims = 2))
    else  # k == 2
        R = right_env(ctr, i, ∂v[(2*j+10):end])
        if ctr.onGPU
            R = CuArray(R)
        end

        eng_loc, eng_l, eng_lu, eng_u, eng_12, plb2, prf2, pd2, splb1, splb2, sprf2, spd2 =
            precompute_conditional(T, ctr, ctr.current_node) #TODO split the line

        eng_l = [@view eng_l[m][:, ∂v[2*j-1+m]] for m ∈ 1:2]
        eng_lu = [@view eng_lu[m][:, ∂v[2*j+1+m]] for m ∈ 1:2]
        eng_u = [@view eng_u[m][:, ∂v[2*j+3+m]] for m ∈ 1:2]
        eng_12 = @view eng_12[∂v[2*j+6], :]

        le =
            eng_loc .+ eng_l[1] .+ eng_l[2] .+ eng_lu[1] .+ eng_lu[2] .+ eng_u[1] .+
            eng_u[2] .+ eng_12
        ele = exp.(-β .* (le .- minimum(le)))

        # @cast LMX5[x, y, v, p1, p2] := LMX[(x, y), (v, p1, p2)] (
        #     p1 ∈ 1:splb1,
        #     p2 ∈ 1:splb2,
        #     x ∈ 1:1,
        # )
        LMX5 = reshape(LMX, 1, size(LMX, 1), size(LMX, 2) ÷ (splb1 * splb2), splb1, splb2)
        LMX3 = LMX5[:, :, ∂v[2*j-1], ∂v[2*j+7], :]
        # @cast M4[x, y, p1, p2] := M[x, y, (p1, p2)] (p2 ∈ 1:spd2)
        M4 = reshape(M, size(M, 1), size(M, 2), size(M, 3) ÷ spd2, spd2)
        M2 = M4[:, :, ∂v[2*j+8], :]
        # @cast R4[x, y, p1, p2] := R[(x, y), (p1, p2)] (p2 ∈ 1:sprf2, y ∈ 1:1)
        R4 = reshape(R, size(R, 1), 1, size(R, 2) ÷ sprf2, sprf2)
        R2 = R4[:, :, ∂v[2*j+9], :]
        LR = dropdims(LMX3[:, :, plb2] ⊠ M2[:, :, pd2] ⊠ R2[:, :, prf2], dims = (1, 2))
        probs = Array(ele .* LR)
    end
    push!(ctr.statistics, ((i, j), ∂v) => error_measure(probs))
    normalize_probability(probs)
end

"""
$(TYPEDSIGNATURES)
Generate the search order of nodes for a matrix product states (MPS).

## Arguments
- `peps::PEPSNetwork{T, S}`: PEPS tensor network with a specific tensor layout.

## Returns
- Tuple{Vector{Tuple{Int, Int, Int}}, Tuple{Int, Int, Int}}: Tuple containing the list of node coordinates and the size of the tensor network.

## Description
The `nodes_search_order_Mps` function generates the search order of nodes for a matrix product states (MPS).
It creates a list of node coordinates `(i, j, k)` representing rows, columns, and index of group of spins, respectively.
The resulting order is suitable for traversing the nodes in the tensor network during contraction.
The function is specialized for the `SquareCrossDoubleNode` tensor network type and is parametrized by the layout type `S`.
"""
function nodes_search_order_Mps(peps::PEPSNetwork{T,S}) where {T<:SquareCrossDoubleNode,S}
    (
        [(i, j, k) for i ∈ 1:peps.nrows for j ∈ 1:peps.ncols for k ∈ 1:2],
        (peps.nrows + 1, 1, 1),
    )
end

"""
$(TYPEDSIGNATURES)
Compute the boundary states for a specific node in a matrix product states (MPS).

## Arguments
- `T::Type`: Tensor network type, specialized for `SquareCrossDoubleNode`.
- `ctr::MpsContractor{S}`: MPS tensor network contractor containing relevant information.
- `node::Node`: Tuple representing the coordinates of the node in the tensor network.

## Returns
- Vector{Tuple{Tuple, Tuple}}: Vector of tuples representing the boundary states for the given node.
Each tuple contains pairs of indices representing connected sites in the tensor network.

## Description
The `boundary` function computes the boundary states for a specific node in a matrix product states (MPS).
The boundary states are determined by analyzing the connections between the current node and its neighboring nodes, considering different physical indices.
The function is specialized for the `SquareCrossDoubleNode` tensor network type and is parametrized by the layout type `S`.
"""
function boundary(
    ::Type{T},
    ctr::MpsContractor{S},
    node::Node,
) where {T<:SquareCrossDoubleNode,S}
    i, j, k = node  # todo
    if k == 1
        bnd = vcat(
            [
                [
                    (
                        (i, m - 1, 1),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m - 1, 2),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m, 1),
                        ((i + 1, m - 1, 1), (i + 1, m - 1, 2)),
                        (i, m, 2),
                        ((i + 1, m - 1, 1), (i + 1, m - 1, 2)),
                    ),
                    (
                        (i, m, 1),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m, 2),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                    ),
                ] for m ∈ 1:(j-1)
            ]...,
            (
                (i, j - 1, 1),
                ((i + 1, j, 1), (i + 1, j, 2)),
                (i, j - 1, 2),
                ((i + 1, j, 1), (i + 1, j, 2)),
            ),
            ((i, j - 1, 1), (i, j, 1)),
            ((i, j - 1, 1), (i, j, 2)),
            ((i, j - 1, 2), (i, j, 1)),
            ((i, j - 1, 2), (i, j, 2)),
            ((i - 1, j - 1, 1), (i, j, 1)),
            ((i - 1, j - 1, 1), (i, j, 2)),
            ((i - 1, j - 1, 2), (i, j, 1)),
            ((i - 1, j - 1, 2), (i, j, 2)),
            ((i - 1, j, 1), (i, j, 1)),
            ((i - 1, j, 1), (i, j, 2)),
            ((i - 1, j, 2), (i, j, 1)),
            ((i - 1, j, 2), (i, j, 2)),
            [
                [
                    (
                        (i - 1, m - 1, 1),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m - 1, 2),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m, 1),
                        ((i, m - 1, 1), (i, m - 1, 2)),
                        (i - 1, m, 2),
                        ((i, m - 1, 1), (i, m - 1, 2)),
                    ),
                    (
                        (i - 1, m, 1),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m, 2),
                        ((i, m, 1), (i, m, 2)),
                    ),
                ] for m ∈ (j+1):ctr.peps.ncols
            ]...,
        )
    else  # k == 2
        bnd = vcat(
            [
                [
                    (
                        (i, m - 1, 1),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m - 1, 2),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m, 1),
                        ((i + 1, m - 1, 1), (i + 1, m - 1, 2)),
                        (i, m, 2),
                        ((i + 1, m - 1, 1), (i + 1, m - 1, 2)),
                    ),
                    (
                        (i, m, 1),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                        (i, m, 2),
                        ((i + 1, m, 1), (i + 1, m, 2)),
                    ),
                ] for m ∈ 1:(j-1)
            ]...,
            (
                (i, j - 1, 1),
                ((i + 1, j, 1), (i + 1, j, 2)),
                (i, j - 1, 2),
                ((i + 1, j, 1), (i + 1, j, 2)),
            ),
            ((i, j - 1, 1), (i, j, 2)),
            ((i, j - 1, 2), (i, j, 2)),
            ((i - 1, j - 1, 1), (i, j, 2)),
            ((i - 1, j - 1, 2), (i, j, 2)),
            ((i - 1, j, 1), (i, j, 2)),
            ((i - 1, j, 2), (i, j, 2)),
            ((i, j, 1), (i, j, 2)),
            ((i, j, 1), ((i + 1, j - 1, 1), (i + 1, j - 1, 2))),
            ((i, j, 1), ((i + 1, j, 1), (i + 1, j, 2))),
            (
                (i, j, 1),
                (
                    (i + 1, j + 1, 1),
                    (i + 1, j + 1, 2),
                    (i, j + 1, 1),
                    (i, j + 1, 2),
                    (i - 1, j + 1, 1),
                    (i - 1, j + 1, 2),
                ),
            ),
            [
                [
                    (
                        (i - 1, m - 1, 1),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m - 1, 2),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m, 1),
                        ((i, m - 1, 1), (i, m - 1, 2)),
                        (i - 1, m, 2),
                        ((i, m - 1, 1), (i, m - 1, 2)),
                    ),
                    (
                        (i - 1, m, 1),
                        ((i, m, 1), (i, m, 2)),
                        (i - 1, m, 2),
                        ((i, m, 1), (i, m, 2)),
                    ),
                ] for m ∈ (j+1):ctr.peps.ncols
            ]...,
        )
    end
    bnd
end

"""
$(TYPEDSIGNATURES)
Update the energy of a specific tensor node in a matrix product states (MPS).

## Arguments
- `T::Type`: Tensor network type, specialized for `SquareCrossDoubleNode`.
- `ctr::MpsContractor{S}`: MPS tensor network contractor containing relevant information for contraction.
- `σ::Vector{Int}`: State vector representing the current configuration of the tensor network.

## Returns
- Real: Updated energy value for the specified tensor node.

## Description
The `update_energy` function calculates the energy contribution of a specific tensor node in a matrix product states (MPS).
The energy is computed based on the local energy at the node and the interaction energies with its neighboring nodes, considering the provided state vector `σ`.
The function is specialized for the `SquareCrossDoubleNode` tensor network type and is parametrized by the layout type `S`.
"""
function update_energy(
    ::Type{T},
    ctr::MpsContractor{S},
    σ::Vector{Int},
) where {T<:SquareCrossDoubleNode,S}
    net = ctr.peps
    i, j, k = ctr.current_node

    en = local_energy(net, (i, j, k))
    for v ∈ (
        (i, j - 1, 1),
        (i, j - 1, 2),
        (i - 1, j, 1),
        (i - 1, j, 2),
        (i - 1, j - 1, 1),
        (i - 1, j - 1, 2),
        (i - 1, j + 1, 1),
        (i - 1, j + 1, 2),
    )
        en += bond_energy(net, (i, j, k), v, local_state_for_node(ctr, σ, v))
    end
    if k != 2
        return en
    end
    en += bond_energy(net, (i, j, k), (i, j, 1), local_state_for_node(ctr, σ, (i, j, 1)))  # here k=2
    en
end

"""
$(TYPEDSIGNATURES)
Generate the tensor for a central double node in a projected entangled pair states (PEPS) tensor network.

## Arguments
- `net::PEPSNetwork{T, Sparse}`: PEPS tensor network.
- `node::PEPSNode`: Node representing the position of the central double node.
- `β::Real`: Inverse temperature parameter.
- `::Val{:central_d_double_node}`: symbol to indicate the central double node.

## Returns
- `DiagonalTensor`: Tensor representing the central double node.

## Description
The `tensor` function generates the tensor for a central double node in a PEPS tensor network.
It uses the inverse temperature parameter `β` to construct the central tensor based on the geometry of the tensor network.
The function is specialized for PEPS tensor networks with sparse tensors (`Sparse`) and is parametrized by the abstract geometry type `T`.
"""
function tensor(
    net::PEPSNetwork{T,Sparse},
    node::PEPSNode,
    β::Real,
    ::Val{:central_d_double_node},
) where {T<:AbstractGeometry}
    i, j = floor(Int, node.i), floor(Int, node.j)
    T_NW_SE = CentralTensor(net, β, (i, j), (i + 1, j + 1))
    T_NE_SW = CentralTensor(net, β, (i, j + 1), (i + 1, j))
    DiagonalTensor(T_NW_SE, T_NE_SW)
end

"""
$(TYPEDSIGNATURES)
Determine the size of the tensor corresponding to a central double node in a projected entangled pair states (PEPS) tensor network.

## Arguments
- `net::PEPSNetwork{SquareCrossDoubleNode{T}, S}`: PEPS tensor network with square cross double nodes.
- `node::PEPSNode`: Node representing the position of the central double node.
- `::Val{:central_d_double_node}`: symbol to indicate the central double node.

## Returns
- `Tuple{Int, Int}`: Tuple representing the size of the tensor for the central double node.

## Description
The `Base.size` function is used to determine the size of the tensor corresponding to a central double node in a PEPS tensor network.
It calculates the size by considering the sizes of the two central tensors associated with neighboring positions.
The function is parametrized by the abstract tensors layout type `T` and the abstract sparsity type `S`.
"""
function Base.size(
    net::PEPSNetwork{SquareCrossDoubleNode{T},S},
    node::PEPSNode,
    ::Val{:central_d_double_node},
) where {T<:AbstractTensorsLayout,S<:AbstractSparsity}
    i, j = floor(Int, node.i), floor(Int, node.j)
    s_1 = CentralTensor_size(net, (i, j), (i + 1, j + 1))
    s_2 = CentralTensor_size(net, (i, j + 1), (i + 1, j))
    (s_1[1] * s_2[1], s_1[2] * s_2[2])
end

"""
$(TYPEDSIGNATURES)
Create a sparse virtual double node tensor in a projected entangled pair states (PEPS) tensor network.

## Arguments
- `net::PEPSNetwork{SquareCrossDoubleNode{T}, S}`: PEPS tensor network with square cross double nodes.
- `node::PEPSNode`: Node representing the position of the virtual double node.
- `β::Real`: Inverse temperature parameter.
- `::Val{:sparse_virtual_double_node}`: symbol to indicate the creation of a sparse virtual double node tensor.

## Returns
- `VirtualTensor{T, S}`: Sparse virtual double node tensor.

## Description
The `tensor` function is used to create a sparse virtual double node tensor in a PEPS tensor network.
It constructs the tensor by incorporating information about the central tensor and surrounding projectors associated with the specified position.
The function is parametrized by the abstract tensors layout type `T`, and the abstract sparsity type `S`, which can be either `Sparse` or `Dense`.

"""
function tensor(  #TODO
    net::PEPSNetwork{SquareCrossDoubleNode{T},S},
    node::PEPSNode,
    β::Real,
    ::Val{:sparse_virtual_double_node},
) where {T<:AbstractTensorsLayout,S<:Union{Sparse,Dense}}
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    p_lb = [projector(net, (i, j, k), ((i + 1, j + 1, 1), (i + 1, j + 1, 2))) for k ∈ 1:2]
    p_l = [projector(net, (i, j, k), ((i, j + 1, 1), (i, j + 1, 2))) for k ∈ 1:2]
    p_lt = [projector(net, (i, j, k), ((i - 1, j + 1, 1), (i - 1, j + 1, 2))) for k ∈ 1:2]

    p_rb = [projector(net, (i, j + 1, k), ((i + 1, j, 1), (i + 1, j, 2))) for k ∈ 1:2]
    p_r = [projector(net, (i, j + 1, k), ((i, j, 1), (i, j, 2))) for k ∈ 1:2]
    p_rt = [projector(net, (i, j + 1, k), ((i - 1, j, 1), (i - 1, j, 2))) for k ∈ 1:2]

    p_lb[1], p_l[1], p_lt[1] = last(fuse_projectors((p_lb[1], p_l[1], p_lt[1])))
    p_lb[2], p_l[2], p_lt[2] = last(fuse_projectors((p_lb[2], p_l[2], p_lt[2])))

    p_rb[1], p_r[1], p_rt[1] = last(fuse_projectors((p_rb[1], p_r[1], p_rt[1])))
    p_rb[2], p_r[2], p_rt[2] = last(fuse_projectors((p_rb[2], p_r[2], p_rt[2])))

    p_lb = outer_projector(p_lb[1], p_lb[2])
    p_l = outer_projector(p_l[1], p_l[2])
    p_lt = outer_projector(p_lt[1], p_lt[2])

    p_rb = outer_projector(p_rb[1], p_rb[2])
    p_r = outer_projector(p_r[1], p_r[2])
    p_rt = outer_projector(p_rt[1], p_rt[2])

    VirtualTensor(
        net.lp,
        CentralTensor(net, β, (i, j), (i, j + 1)),
        (p_lb, p_l, p_lt, p_rb, p_r, p_rt),
    )
end

"""
$(TYPEDSIGNATURES)

Create a dense virtual double node tensor in a projected entangled pair states (PEPS) tensor network.

## Arguments
- `net::PEPSNetwork{T, Dense}`: PEPS tensor network with nodes of type `T` and dense tensors.
- `node::PEPSNode`: Node representing the position of the virtual double node.
- `β::Real`: Inverse temperature parameter.
- `::Val{:virtual_double_node}`: symbol to indicate the creation of a dense virtual double node tensor.

## Returns
- `Tensor{T}`: Dense virtual double node tensor.

## Description
The `tensor` function is used to create a dense virtual double node tensor in a PEPS tensor network.
It constructs the tensor by combining information from the sparse virtual double node tensor, including
projectors and the dense central tensor associated with the specified position.
The function is parametrized by the abstract geometry type `T`.
"""
function tensor(
    net::PEPSNetwork{T,Dense},
    node::PEPSNode,
    β::Real,
    ::Val{:virtual_double_node},
) where {T<:AbstractGeometry}
    sp = tensor(net, node, β, Val(:sparse_virtual_double_node))
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = (get_projector!(net.lp, x) for x in sp.projs)
    dense_con = dense_central_tensor(sp.con)

    A = zeros(
        eltype(dense_con),
        length(p_l),
        maximum.((p_lt, p_rt))...,
        length(p_r),
        maximum.((p_lb, p_rb))...,
    )
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        A[l, p_lt[l], p_rt[r], r, p_lb[l], p_rb[r]] = dense_con[p_l[l], p_r[r]]
    end
    # @cast B[l, (uu, u), r, (dd, d)] := A[l, uu, u, r, dd, d]
    l, uu, u, r, dd, d = size(A)
    B = reshape(A, l, uu * u, r, dd * d)
    B
end

"""
$(TYPEDSIGNATURES)

Construct the set of projectors associated with a site tensor in a projected entangled pair states (PEPS) tensor network.

## Arguments
- `net::PEPSNetwork{T, S}`: PEPS tensor network with nodes of type `T` and tensors of sparsity type `S`.
- `vertex::Node`: Node representing the position of the site tensor.

## Returns
- `(plf, pt, prf, pb)`: Tuple of projectors associated with the site tensor, corresponding to the left (plf), top (pt), right (prf), and bottom (pb) directions.

## Description
The `projectors_site_tensor` function constructs the set of projectors associated with a site tensor at the specified position in a PEPS tensor network.
The projectors are created based on the neighboring tensors and directions in the network.
The function is parametrized by the abstract node type `T` and the abstract sparsity type `S`.
"""
function projectors_site_tensor(
    net::PEPSNetwork{T,S},
    vertex::Node,
) where {T<:SquareCrossDoubleNode,S}
    i, j = vertex
    plf = outer_projector(
        (
            projector(
                net,
                (i, j, k),
                (
                    (i + 1, j - 1, 1),
                    (i + 1, j - 1, 2),
                    (i, j - 1, 1),
                    (i, j - 1, 2),
                    (i - 1, j - 1, 1),
                    (i - 1, j - 1, 2),
                ),
            ) for k ∈ 1:2
        )...,
    )
    pt = outer_projector(
        (projector(net, (i, j, k), ((i - 1, j, 1), (i - 1, j, 2))) for k ∈ 1:2)...,
    )
    prf = outer_projector(
        (
            projector(
                net,
                (i, j, k),
                (
                    (i + 1, j + 1, 1),
                    (i + 1, j + 1, 2),
                    (i, j + 1, 1),
                    (i, j + 1, 2),
                    (i - 1, j + 1, 1),
                    (i - 1, j + 1, 2),
                ),
            ) for k ∈ 1:2
        )...,
    )
    pb = outer_projector(
        (projector(net, (i, j, k), ((i + 1, j, 1), (i + 1, j, 2))) for k ∈ 1:2)...,
    )
    (plf, pt, prf, pb)
end

"""
$(TYPEDSIGNATURES)

Determine the size of the virtual tensor associated with a virtual double node in a tensor network.

## Arguments
- `net::AbstractGibbsNetwork{Node, PEPSNode}`: Abstract Gibbs PEPS tensor network with nodes and virtual tensors.
- `node::PEPSNode`: PEPS node representing the position of the virtual double node.
- `::Union{Val{:virtual_double_node}, Val{:sparse_virtual_double_node}}`: Tag indicating whether the virtual tensor is dense or sparse.

## Returns
- Tuple `(s1, s2, s3, s4)`: Size information for the virtual tensor.

## Description
The `size` function determines the size of the virtual tensor associated with a virtual double node in an Abstract Gibbs PEPS tensor network.
The size is specified by the dimensions along the left, top, right, and bottom directions.
The function is parametrized by the types of nodes (`Node` and `PEPSNode`) and the tag indicating whether the virtual tensor is dense or sparse.
"""
function Base.size(
    net::AbstractGibbsNetwork{Node,PEPSNode},
    node::PEPSNode,
    ::Union{Val{:virtual_double_node},Val{:sparse_virtual_double_node}},
)
    v = Node(node)
    i, j = node.i, floor(Int, node.j)

    p_lb = [projector(net, (i, j, k), ((i + 1, j + 1, 1), (i + 1, j + 1, 2))) for k ∈ 1:2]
    p_l = [projector(net, (i, j, k), ((i, j + 1, 1), (i, j + 1, 2))) for k ∈ 1:2]
    p_lt = [projector(net, (i, j, k), ((i - 1, j + 1, 1), (i - 1, j + 1, 2))) for k ∈ 1:2]

    p_rb = [projector(net, (i, j + 1, k), ((i + 1, j, 1), (i + 1, j, 2))) for k ∈ 1:2]
    p_r = [projector(net, (i, j + 1, k), ((i, j, 1), (i, j, 2))) for k ∈ 1:2]
    p_rt = [projector(net, (i, j + 1, k), ((i - 1, j, 1), (i - 1, j, 2))) for k ∈ 1:2]

    p_lb[1], p_l[1], p_lt[1] = last(fuse_projectors((p_lb[1], p_l[1], p_lt[1])))
    p_lb[2], p_l[2], p_lt[2] = last(fuse_projectors((p_lb[2], p_l[2], p_lt[2])))

    p_rb[1], p_r[1], p_rt[1] = last(fuse_projectors((p_rb[1], p_r[1], p_rt[1])))
    p_rb[2], p_r[2], p_rt[2] = last(fuse_projectors((p_rb[2], p_r[2], p_rt[2])))

    p_lb = outer_projector(p_lb[1], p_lb[2])
    p_l = outer_projector(p_l[1], p_l[2])
    p_lt = outer_projector(p_lt[1], p_lt[2])

    p_rb = outer_projector(p_rb[1], p_rb[2])
    p_r = outer_projector(p_r[1], p_r[2])
    p_rt = outer_projector(p_rt[1], p_rt[2])

    (
        size(p_l, 1),
        maximum(p_lt) * maximum(p_rt),
        size(p_r, 1),
        maximum(p_lb) * maximum(p_rb),
    )
end
