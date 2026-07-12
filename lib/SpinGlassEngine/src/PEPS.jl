# PEPS.jl: This file provides all tools needed to build PEPS tensor network.

using LabelledGraphs

export AbstractGibbsNetwork,
    interaction_energy,
    connecting_tensor,
    normalize_probability,
    fuse_projectors,
    initialize_gauges!,
    decode_state,
    PEPSNetwork,
    mod_wo_zero,
    bond_energy,
    outer_projector,
    projector,
    spectrum,
    is_compatible,
    ones_like,
    _equalize,
    _normalize,
    branch_solution,
    local_spins,
    local_energy

# T: type of the vertex of network
# S: type of the vertex of underlying factor graph
abstract type AbstractGibbsNetwork{S,T,R} end

"""
$(TYPEDSIGNATURES)

Construct a Projected Entangled Pair States (PEPS) network.

# Arguments
- `m::Int`: Number of rows in the PEPS lattice.
- `n::Int`: Number of columns in the PEPS lattice.
- `potts_hamiltonian::LabelledGraph`: Potts Hamiltonian representing the Hamiltonian.
- `transformation::LatticeTransformation`: Transformation of the PEPS lattice, as it can be rotated or reflected. 
- `gauge_type::Symbol=:id`: Type of gauge to initialize (default is identity).

# Type Parameters
- `T <: AbstractGeometry`: Type of node used within the PEPS tensor network. It can be `SquareSingleNode`, `SquareDoubleNode`, `KingSingleNode`, `SquareCrossDoubleNode`.
- `S <: AbstractSparsity`: Type of sparsity for the PEPS tensors: `Dense` or `Sparse`.
- `R <: Real``: The numeric precision type for real values (e.g., Float64).

# Returns
An instance of PEPSNetwork{T, S, R}.
"""
mutable struct PEPSNetwork{T<:AbstractGeometry,S<:AbstractSparsity,R<:Real} <:
               AbstractGibbsNetwork{Node,PEPSNode,R}
    potts_hamiltonian::LabelledGraph
    vertex_map::Function
    lp::PoolOfProjectors
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensors_map::Dict{PEPSNode,Symbol}
    gauges::Gauges{T}

    function PEPSNetwork{T,S,R}(
        m::Int,
        n::Int,
        potts_hamiltonian::LabelledGraph,
        transformation::LatticeTransformation,
        gauge_type::Symbol = :id,
    ) where {T<:AbstractGeometry,S<:AbstractSparsity,R<:Real}
        lp = get_prop(potts_hamiltonian, :pool_of_projectors)
        net = new(potts_hamiltonian, vertex_map(transformation, m, n), lp, m, n)
        net.nrows, net.ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(net.potts_hamiltonian, T.name.wrapper(m, n))
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        net.tensors_map = tensor_map(T, S, net.nrows, net.ncols)
        net.gauges = Gauges{T,R}(net.nrows, net.ncols)
        initialize_gauges!(net, gauge_type)
        net
    end
end

"""
$(TYPEDSIGNATURES)

Calculate the modulo operation of `k` with respect to `m`, ensuring the result is not zero.

## Arguments
- `k`: The dividend.
- `m`: The divisor.

## Returns
- `result::Int`: The result of `k % m`, ensuring it is not zero.
"""
mod_wo_zero(k, m) = k % m == 0 ? m : k % m

"""
$(TYPEDSIGNATURES)
Create an identity with the same type as the input number `x`.

## Arguments
- `x`: A numeric value.

## Returns
- a multiplicative identity with the same type as `x`.
"""
ones_like(x::Number) = one(typeof(x))

"""
$(TYPEDSIGNATURES)

Create an array of ones with the same element type and size as the input array `x`.

## Arguments
- `x::AbstractArray`: An array serving as a template.

## Returns
- `result::Array`: An array of ones with the same element type and size as `x`.
"""
ones_like(x::AbstractArray) = ones(eltype(x), size(x))

"""
$(TYPEDSIGNATURES)

Calculate the bond energy between nodes `u` and `v` for a given index `σ` in the Gibbs network `net`.

## Arguments
- `net::AbstractGibbsNetwork{T, S}`: The Gibbs network.
- `u::Node`: One of the nodes connected by the bond.
- `v::Node`: The other node connected by the bond.
- `σ::Int`: The index for which the bond energy is calculated.

## Returns
- `energies::Vector{T}`: Vector containing the bond energies between nodes `u` and `v` for index `σ`.
"""
function bond_energy(
    net::AbstractGibbsNetwork{T,S,R},
    u::Node,
    v::Node,
    σ::Int,
) where {T,S,R}
    potts_h_u, potts_h_v = net.vertex_map(u), net.vertex_map(v)
    energies = SpinGlassNetworks.bond_energy(net.potts_hamiltonian, potts_h_u, potts_h_v, σ)
    R.(vec(energies))
end

"""
$(TYPEDSIGNATURES)

Compute the projector between two nodes `v` and `w` in the Gibbs network `network`.

## Arguments
- `network::AbstractGibbsNetwork{S, T}`: The Gibbs network.
- `v::S`: Source node.
- `w::S`: Target node.

## Returns
- `projector::Matrix{T}`: Projector matrix between nodes `v` and `w`.
"""
function projector(network::AbstractGibbsNetwork{S,T}, v::S, w::S) where {S,T}
    potts_h = network.potts_hamiltonian
    potts_h_v, potts_h_w = network.vertex_map(v), network.vertex_map(w)
    SpinGlassNetworks.projector(potts_h, potts_h_v, potts_h_w)
end

"""
$(TYPEDSIGNATURES)

Compute the projector matrix for the given node `v` onto a tuple of target nodes `vertices` in the Gibbs network `net`.

## Arguments 
- `net::AbstractGibbsNetwork{S, T}`: The Gibbs network.
- `v::S`: Source node.
- `vertices::NTuple{N, S}`: Tuple of target nodes onto which the projector is computed.
    
## Returns
- first fused projector matrix for node `v` onto the specified target nodes.
    
"""
function projector(
    net::AbstractGibbsNetwork{S,T},
    v::S,
    vertices::NTuple{N,S},
) where {S,T,N}
    first(fuse_projectors(projector.(Ref(net), Ref(v), vertices)))
end

"""
$(TYPEDSIGNATURES)
Fuse a tuple of projector matrices into a single projector matrix using rank-revealing techniques.

## Arguments
- `projectors::NTuple{N, K}`: Tuple of projector matrices to be fused.

## Returns
- `fused::Matrix{Float64}`: Fused projector matrix.
- `transitions::NTuple{N, Vector{Int}}`: Tuple of transition vectors indicating the indices of the non-zero rows in each original projector.
"""
function fuse_projectors(
    projectors::NTuple{N,K},
    #projectors::Union{Vector{S}, NTuple{N, S}}
) where {N,K}
    fused, transitions_matrix = rank_reveal(hcat(projectors...), :PE)
    # transitions = collect(eachcol(transitions_matrix))
    transitions = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    fused, transitions
end

function outer_projector(p1::Array{T,1}, p2::Array{T,1}) where {T<:Number}
    reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
end

"""
$(TYPEDSIGNATURES)
Retrieve the spectrum associated with a specific vertex in the Gibbs network.

## Arguments
- `network::AbstractGibbsNetwork{S, T}`: Gibbs network containing the Potts Hamiltonian.
- `vertex::S`: Vertex for which the spectrum is to be retrieved.

## Returns
- Spectrum associated with the specified vertex.
"""
function spectrum(network::AbstractGibbsNetwork{S,T}, vertex::S) where {S,T}
    get_prop(network.potts_hamiltonian, network.vertex_map(vertex), :spectrum)
end

"""
$(TYPEDSIGNATURES)
Retrieve the local energy spectrum associated with a specific vertex in the Gibbs network.

## Arguments
- `network::AbstractGibbsNetwork{S, T}`: Gibbs network containing the Potts Hamiltonian.
- `vertex::S`: Vertex for which the local energy spectrum is to be retrieved.

## Returns
- Local energy spectrum associated with the specified vertex.
"""
function local_energy(network::AbstractGibbsNetwork{S,T,R}, vertex::S) where {S,T,R}
    R.(spectrum(network, vertex).energies)
end


"""
$(TYPEDSIGNATURES)
Determine the cluster size associated with a specific vertex in the Gibbs network.

## Arguments
- `net::AbstractGibbsNetwork{S, T}`: Gibbs network containing the Potts Hamiltonian.
- `v::S`: Vertex for which the cluster size is to be determined.

## Returns
- `size::Int`: Number of states in the local energy spectrum associated with the specified vertex.
"""
function SpinGlassNetworks.cluster_size(net::AbstractGibbsNetwork{S,T}, v::S) where {S,T}
    length(local_energy(net, v))
end

"""
$(TYPEDSIGNATURES)
Compute the interaction energy between two vertices in a Gibbs network.

## Arguments
- `network::AbstractGibbsNetwork{S, T}`: Gibbs network containing the Potts Hamiltonian.
- `v::S`: First vertex.
- `w::S`: Second vertex.

## Returns
- `energy::Matrix{T}`: Interaction energy matrix between vertices `v` and `w`.
"""
function interaction_energy(network::AbstractGibbsNetwork{S,T,R}, v::S, w::S) where {S,T,R}
    potts_h = network.potts_hamiltonian
    potts_h_v, potts_h_w = network.vertex_map(v), network.vertex_map(w)
    if has_edge(potts_h, potts_h_w, potts_h_v)
        R.(get_prop(potts_h, potts_h_w, potts_h_v, :en)')
    elseif has_edge(potts_h, potts_h_v, potts_h_w)
        R.(get_prop(potts_h, potts_h_v, potts_h_w, :en))
    else
        zeros(R, 1, 1)
    end
end

"""
$(TYPEDSIGNATURES)
Check if a Potts Hamiltonian is compatible with a given network graph.

## Arguments
- `potts_hamiltonian::LabelledGraph`: Graph representing the Potts Hamiltonian.
- `network_graph::LabelledGraph`: Graph representing the network.
    
## Returns
- `compatibility::Bool`: `true` if the Potts Hamiltonian is compatible with the network graph, `false` otherwise.
"""
function is_compatible(potts_hamiltonian::LabelledGraph, network_graph::LabelledGraph)
    all(has_edge(network_graph, src(edge), dst(edge)) for edge ∈ edges(potts_hamiltonian))
end

"""
$(TYPEDSIGNATURES)
Initialize gauge tensors in a Gibbs network.

## Arguments
- `net::AbstractGibbsNetwork{S, T}`: Gibbs network to initialize.
- `type::Symbol=:id`: Type of initialization, either `:id` for identity or `:rand` for random values.

## Description
This function initializes gauge tensors in a Gibbs network according to the specified type. 
Each gauge tensor is associated with two positions in the network and a type. 
The positions are determined by the gauge's `positions` field, and the type is specified by the gauge's `type` field. 
The initialization type can be either `:id` for identity tensors or `:rand` for random tensors.
"""
function initialize_gauges!(
    net::AbstractGibbsNetwork{S,T,R},
    type::Symbol = :id,
) where {S,T,R}
    @assert type ∈ (:id, :rand)
    for gauge ∈ net.gauges.info
        n1, n2 = gauge.positions
        push!(net.tensors_map, n1 => gauge.type, n2 => gauge.type)
        d = size(net, gauge.attached_tensor)[gauge.attached_leg]
        X = type == :id ? ones(R, d) : rand(R, d) .+ 0.42
        push!(net.gauges.data, n1 => X, n2 => 1 ./ X)
    end
end

"""
$(TYPEDSIGNATURES)
Normalize a probability distribution.

## Arguments
- `probs::Vector{<:Real}`: A vector representing a probability distribution.

## Returns
- `Vector{Float64}`: Normalized probability distribution.
"""
_normalize(probs::Vector{<:Real}) = probs ./ sum(probs)

"""
$(TYPEDSIGNATURES)
Equalize a probability distribution.

## Arguments
- `probs::Vector{<:Real}`: A vector representing a probability distribution.

## Returns
- `Vector{Float64}`: Equalized probability distribution.
"""
function _equalize(probs::Vector{<:Real})
    mp = abs(minimum(probs))
    _normalize(replace(p -> p < mp ? mp : p, probs))
end

"""
$(TYPEDSIGNATURES)
Normalize a probability distribution.

## Arguments
- `probs::Vector{<:Real}`: A vector representing a probability distribution.

## Returns
- `Vector{Float64}`: Normalized probability distribution.
"""
function normalize_probability(probs::Vector{<:Real})
    if minimum(probs) < 0
        return _equalize(probs)
    end
    _normalize(probs)
end

"""
$(TYPEDSIGNATURES)
Decode a state vector into a dictionary representation.

## Arguments
- `peps::AbstractGibbsNetwork{S, T}`: The Gibbs network.
- `σ::Vector{Int}`: State vector to be decoded.
- `potts_h_order::Bool=false`: If true, use the order of nodes in the Potts Hamiltonian.

## Returns
- `Dict{Symbol, Int}`: A dictionary mapping node symbols to corresponding values in the state vector.
"""
function decode_state(
    peps::AbstractGibbsNetwork{S,T},
    σ::Vector{Int},
    potts_h_order::Bool = false,
) where {S,T}
    nodes =
        potts_h_order ? peps.vertex_map.(nodes_search_order_Mps(peps)) :
        vertices(peps.potts_hamiltonian)
    Dict(nodes[1:length(σ)] .=> σ)
end
