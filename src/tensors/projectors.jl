export PoolOfProjectors, get_projector!, add_projector!, empty!

const Proj{T} = Union{Vector{T},CuArray{T,1}}

# Cached result of merging a projector triple (see merge_projectors_inter):
# (fused-triple projector, fused-pair projector, pair rank).
const MergedProj{T} = Tuple{Proj{T},Proj{T},Int}

"""
$(TYPEDSIGNATURES)

`PoolOfProjectors` is a data structure for managing projectors associated with Ising model sites.
It allows efficient storage and retrieval of projectors based on their indices and provides support for different computational devices.

# Fields:
- `data::Dict{Symbol, Dict{Int, Proj{T}}}`: A dictionary that stores projectors associated with different
computational devices (`:CPU`, `:GPU`, etc.). The inner dictionary maps site indices to projectors.
- `default_device::Symbol`: A symbol representing the default computational device for projectors in the pool.
- `sizes::Dict{Int, Int}`: A dictionary that maps site indices to the maximum projector size for each site.

# Constructors:
- `PoolOfProjectors(data::Dict{Int, Dict{Int, Vector{T}}}) where T`: Create a `PoolOfProjectors` with initial data for projectors.
The data is provided as a dictionary that maps site indices to projectors stored in different computational devices.
The `sizes` dictionary is automatically populated based on the maximum projector size for each site.
- `PoolOfProjectors{T}() where T`: Create an empty `PoolOfProjectors` with no projectors initially stored.
"""
struct PoolOfProjectors{T<:Integer}
    data::Dict{Symbol,Dict{Int,Proj{T}}}
    default_device::Symbol
    sizes::Dict{Int,Int}
    # Cached projector indicator matrices (see projector_matrix): sparse
    # CSC/CuSparseCSR matrices previously memoized process-globally behind
    # pirated SparseArrays.sparse methods. Keys end with the device Symbol.
    matrices::Dict{Tuple,Any}
    # Merged-projector registry: the rank_reveal + index fusion for a triple of
    # projectors is expensive and was recomputed by every VirtualTensor kernel
    # invocation; results are tiny and depend only on (p1, p2, p3, order, device).
    merged::Dict{NTuple{5,Any},MergedProj{T}}

    PoolOfProjectors{T}(data, default_device, sizes) where {T} =
        new{T}(data, default_device, sizes, Dict{Tuple,Any}(), Dict{NTuple{5,Any},MergedProj{T}}())

    PoolOfProjectors(data::Dict{Int,Dict{Int,Vector{T}}}) where {T} = new{T}(
        Dict(:CPU => data),
        :CPU,
        Dict{Int,Int}(k => maximum(v) for (k, v) ∈ data),
        Dict{Tuple,Any}(),
        Dict{NTuple{5,Any},MergedProj{T}}(),
    )
    PoolOfProjectors{T}() where {T} = new{T}(
        Dict(:CPU => Dict{Int,Proj{T}}()),
        :CPU,
        Dict{Int,Int}(),
        Dict{Tuple,Any}(),
        Dict{NTuple{5,Any},MergedProj{T}}(),
    )
end


"""
    copy(lp::PoolOfProjectors)

Create an independent projector workspace. The default-device dictionary is
copied while its projector vectors are shared as read-only definitions;
derived device data and contraction caches start empty. This lets separate
PEPS networks share the same Hamiltonian without duplicating potentially large
definition vectors, racing on mutable caches, or clearing one another's GPU
data.
"""
function Base.copy(lp::PoolOfProjectors{T}) where {T<:Integer}
    definitions = copy(lp.data[lp.default_device])
    data = Dict{Symbol,Dict{Int,Proj{T}}}(lp.default_device => definitions)
    PoolOfProjectors{T}(data, lp.default_device, copy(lp.sizes))
end


Base.eltype(lp::PoolOfProjectors{T}) where {T} = T
Base.length(lp::PoolOfProjectors) = length(lp.data[lp.default_device])
Base.length(lp::PoolOfProjectors, device::Symbol) = length(lp.data[device])

"""
$(TYPEDSIGNATURES)

Empty the pool of projectors associated with a specific computational device.

This function removes all projectors stored on the specified computational device, freeing up memory resources.

# Arguments:
- `lp::PoolOfProjectors`: The `PoolOfProjectors` object containing projectors.
- `device::Symbol`: The computational device for which projectors should be emptied (e.g., `:CPU`, `:GPU`).
"""
function Base.empty!(lp::PoolOfProjectors, device::Symbol)
    if device ∈ keys(lp.data)
        empty!(lp.data[device])
    end
    filter!(kv -> kv.first[5] != device, lp.merged)
    filter!(kv -> kv.first[end] != device, lp.matrices)
    lp
end

Base.length(lp::PoolOfProjectors, index::Int) = length(lp.data[lp.default_device][index])
Base.size(lp::PoolOfProjectors, index::Int) = lp.sizes[index]

get_projector!(lp::PoolOfProjectors, index::Int) =
    get_projector!(lp, index, lp.default_device)

"""
$(TYPEDSIGNATURES)

TODO This is version for only one GPU

Retrieve or create a projector from the `PoolOfProjectors` associated with a specific device.

This function retrieves a projector from the `PoolOfProjectors` if it already exists.
If the projector does not exist in the pool, it creates a new one and stores it for future use on the specified computational device.

# Arguments:
- `lp::PoolOfProjectors{T}`: The `PoolOfProjectors` object containing projectors.
- `index::Int`: The index of the projector to retrieve or create.
- `device::Symbol`: The computational device on which the projector should be stored or retrieved (e.g., `:CPU`, `:GPU`).

# Returns:
- `Proj{T}`: The projector of type `T` associated with the specified index and device.
"""
function get_projector!(
    lp::PoolOfProjectors{T},
    index::Int,
    device::Symbol,
) where {T<:Integer}
    if device ∉ keys(lp.data)
        push!(lp.data, device => Dict{Int,Proj{T}}())
    end

    if index ∉ keys(lp.data[device])
        if device == :GPU
            p = CuArray{T}(lp.data[lp.default_device][index])
        elseif device == :CPU
            p = Array{T}(lp.data[lp.default_device][index])
        else
            throw(ArgumentError("device should be :CPU or :GPU"))
        end
        push!(lp.data[device], index => p)
    end
    lp.data[device][index]
end

"""
$(TYPEDSIGNATURES)

Add a projector to the `PoolOfProjectors` and associate it with an index.

This function adds a projector `p` to the `PoolOfProjectors`.
The `PoolOfProjectors` stores projectors based on their computational device (e.g., CPU or GPU) and assigns a unique index to each projector.
The index can be used to retrieve the projector later using `get_projector!`.

# Arguments:
- `lp::PoolOfProjectors{T}`: The `PoolOfProjectors` object to which the projector should be added.
- `p::Proj`: The projector to be added to the pool. The type of the projector `Proj` should match the type `T` specified in the `PoolOfProjectors`.

# Returns:
- `Int`: The unique index associated with the added projector in the pool.
"""
function add_projector!(lp::PoolOfProjectors{T}, p::Proj) where {T<:Integer}
    if lp.default_device == :CPU
        p = Array{T}(p)
    elseif lp.default_device == :GPU
        p = CuArray{T}(p)
    else
        throw(ArgumentError("default_device should be :CPU or :GPU"))
    end
    if p in values(lp.data[lp.default_device])
        key = -1
        for guess in keys(lp.data[lp.default_device])
            if lp.data[lp.default_device][guess] == p
                key = guess
                break
            end
        end
    else
        key = length(lp.data[lp.default_device]) + 1
        push!(lp.data[lp.default_device], key => p)
        push!(lp.sizes, key => maximum(p))
    end
    key
end
