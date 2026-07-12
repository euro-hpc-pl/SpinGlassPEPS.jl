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
