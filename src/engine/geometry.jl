# geometry.jl: This file provides basic stuctures and types needed for constructing PEPS tensor 
#              network.

export site,
    Node,
    PEPSNode,
    AbstractGeometry,
    AbstractSparsity,
    AbstractTensorsLayout,
    Dense,
    Sparse,
    Gauges,
    GaugeInfo,
    GaugesEnergy,
    EnergyGauges,
    EngGaugesEng,
    SuperPEPSNode

abstract type AbstractGeometry end
abstract type AbstractSparsity end
abstract type AbstractTensorsLayout end

struct Dense <: AbstractSparsity end
struct Sparse <: AbstractSparsity end

struct GaugesEnergy{T} <: AbstractTensorsLayout end
struct EnergyGauges{T} <: AbstractTensorsLayout end
struct EngGaugesEng{T} <: AbstractTensorsLayout end

# ---------------------------------------------------------------------------
# The geometry protocol.
#
# A lattice geometry (a subtype of AbstractGeometry, e.g. SquareSingleNode)
# must implement, for every tensor layout it supports:
#
#   GEOMETRY(m, n)                         -> LabelledGraph  (network graph)
#   tensor_map(GEOMETRY, SPARSITY, nrows, ncols) -> Dict{PEPSNode,Symbol}
#   gauges_list(GEOMETRY{LAYOUT}, nrows, ncols)  -> Vector{GaugeInfo}
#   MpoLayers(GEOMETRY{LAYOUT}, ncols)           -> MpoLayers
#   conditional_probability(GEOMETRY, ctr, boundary_config) -> probabilities
#   projectors_site_tensor(net, vertex)          -> projector tuple
#   nodes_search_order_Mps(net)                  -> (search order, outside node)
#   boundary(GEOMETRY, ctr, node)                -> boundary recipe
#   update_energy(GEOMETRY, ctr, config, node)   -> energy updates
#   Base.size / tensor methods for each tensor species it registers in
#   tensor_map (dispatched via Val(species)).
#
# test/geometry_protocol.jl asserts conformance for every geometry x layout
# combination the solver supports; extend the table there when adding a
# geometry.
# ---------------------------------------------------------------------------

const Node = NTuple{N,Int} where {N}

# """
# $(TYPEDSIGNATURES)
# """
@inline site(i::Site) = denominator(i) == 1 ? numerator(i) : i

"""
$(TYPEDSIGNATURES)

Node for the SquareSingleNode and KingSingleNode.
"""
struct PEPSNode
    i::Site
    j::Site
    PEPSNode(i::Site, j::Site) = new(site(i), site(j))
end
Node(node::PEPSNode) = (node.i, node.j)

"""
$(TYPEDSIGNATURES)

Node for the Pegasus type.
"""
struct SuperPEPSNode
    i::Site
    j::Site
    k::Int

    SuperPEPSNode(i::Site, j::Site, k::Int) = new(site(i), site(j), k)
end
Node(node::SuperPEPSNode) = (node.i, node.j, node.k)

"""
$(TYPEDSIGNATURES)

Defines information how to create gauges.
"""
struct GaugeInfo
    positions::NTuple{2,PEPSNode}
    attached_tensor::PEPSNode
    attached_leg::Int
    type::Symbol
end

"""
$(TYPEDSIGNATURES)

Stores gauges and corresponding information.
"""
struct Gauges{T<:AbstractGeometry,R<:Real}
    data::Dict{PEPSNode,AbstractArray{R}}
    info::Vector{GaugeInfo}

    function Gauges{T,R}(nrows::Int, ncols::Int) where {T<:AbstractGeometry,R<:Real}
        new(Dict{PEPSNode,AbstractArray{R}}(), gauges_list(T, nrows, ncols))
    end
end
