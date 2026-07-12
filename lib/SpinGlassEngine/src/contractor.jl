export SVDTruncate,
    Zipper,
    MpoLayers,
    MpsParameters,
    MpsContractor,
    NoUpdate,
    GaugeStrategy,
    GaugeStrategyWithBalancing,
    clear_memoize_cache,
    clear_memoize_cache_after_row,
    mpo,
    mps_top,
    mps,
    mps_top_approx,
    mps_approx,
    update_gauges!,
    sweep_gauges!,
    update_gauges_with_balancing!,
    boundary_states,
    dressed_mps,
    conditional_probability,
    update_energy,
    boundary,
    local_state_for_node,
    boundary_indices,
    layout,
    sparsity,
    strategy,
    left_env,
    right_env

abstract type AbstractContractor end
abstract type AbstractStrategy end
abstract type AbstractGauge end

struct SVDTruncate <: AbstractStrategy end
struct Zipper <: AbstractStrategy end
struct GaugeStrategyWithBalancing <: AbstractGauge end
struct GaugeStrategy <: AbstractGauge end
struct NoUpdate <: AbstractGauge end

"""
$(TYPEDSIGNATURES)

A struct representing different layers of a Matrix Product Operator (MPO) used in contraction algorithms.

# Fields
- `main::Dict{Site, Sites}`: A dictionary mapping sites to the main layers of the MPO.
- `dress::Dict{Site, Sites}`: A dictionary mapping sites to the dress layers of the MPO.
- `right::Dict{Site, Sites}`: A dictionary mapping sites to the right layers of the MPO.

The `MpoLayers` struct distinguishes the various layers of an MPO, which is often used in tensor network contraction algorithms. MPOs are commonly employed in quantum many-body physics and condensed matter physics to represent operators acting on quantum states in a factorized form.
"""
struct MpoLayers
    main::Dict{Site,Sites}
    dress::Dict{Site,Sites}
    right::Dict{Site,Sites}
end

"""
$(TYPEDSIGNATURES)

A struct representing control parameters for the MPO-MPS (Matrix Product Operator - Matrix Product State) scheme used to contract a PEPS (Projected Entangled Pair States) network.

# Fields
- `bond_dimension::Int`: The maximum bond dimension to be used during contraction.
- `variational_tol::Real`: The tolerance for the variational solver used in MPS optimization. It gives the condition for overlap convergence during one sweep in boundary MPS. Default is 1E-8.
- `max_num_sweeps::Int`: The maximum number of sweeps to perform during variational compression. Default is 4.
- `tol_SVD::Real`: The tolerance used in singular value decomposition (SVD) operations. It means that smaller singular values are truncated. Default is 1E-16.
- `iters_svd::Int`: The number of iterations to perform in SVD computations. Default is 1.
- `iters_var::Int`: The number of iterations for variational optimization. Default is 1.
- `Dtemp_multiplier::Int`: A multiplier for the bond dimension when temporary bond dimensions are computed. Default is 2.
- `method::Symbol`: The type of SVD method to use (e.g., `:psvd_sparse`). Default is `:psvd_sparse`.

Keyword Arguments:
- `bond_dim`: Specifies the maximum bond dimension (default is typemax(Int)).
- `var_tol`: Tolerance for the variational solver (default is 1E-8).
- `num_sweeps`: Maximum number of sweeps for variational compression (default is 4).
- `tol_SVD`: Tolerance for SVD operations (default is 1E-16).
- `iters_svd`: Number of SVD iterations (default is 1).
- `iters_var`: Number of iterations for variational optimization (default is 1).
- `Dtemp_multiplier`: Multiplier for temporary bond dimensions (default is 2).
- `method`: SVD method to use, such as :psvd_sparse (default is :psvd_sparse).

Description:
The MpsParameters struct encapsulates various control parameters that influence the behavior and accuracy of the MPO-MPS contraction scheme used in PEPS network calculations. This allows fine-tuning of tolerances, iteration limits, and methods for efficient and accurate tensor network contractions.
"""
struct MpsParameters{S<:Real}
    bond_dimension::Int
    variational_tol::S
    max_num_sweeps::Int
    tol_SVD::S
    iters_svd::Int
    iters_var::Int
    Dtemp_multiplier::Int
    method::Symbol

    MpsParameters{S}(;
        bond_dim = typemax(Int),
        var_tol::S = S(1E-8),
        num_sweeps = 4,
        tol_SVD::S = S(1E-16),
        iters_svd = 1,
        iters_var = 1,
        Dtemp_multiplier = 2,
        method = :psvd_sparse,
    ) where {S} = new(
        bond_dim,
        var_tol,
        num_sweeps,
        tol_SVD,
        iters_svd,
        iters_var,
        Dtemp_multiplier,
        method,
    )
end

"""
$(TYPEDSIGNATURES)
A function that provides the layout used to construct the PEPS (Projected Entangled Pair States) network.

# Arguments
- `net::PEPSNetwork{T, S}`: The PEPS network for which the layout is provided.

# Returns
- The layout type `T` used to construct the PEPS network.

The `layout` function returns the layout type used in the construction of a PEPS network. This layout type specifies the geometric arrangement and sparsity pattern of the tensors in the PEPS network.
"""
layout(net::PEPSNetwork{T,S}) where {T,S} = T

"""
$(TYPEDSIGNATURES)
A function that provides the sparsity used to construct the PEPS (Projected Entangled Pair States) network.

# Arguments
- `net::PEPSNetwork{T, S}`: The PEPS network for which the sparsity is provided.

# Returns
- The sparsity type `S` used to construct the PEPS network.

The `sparsity` function returns the sparsity type used in the construction of a PEPS network. This sparsity type specifies the pattern of zero elements in the tensors of the PEPS network, which can affect the computational efficiency and properties of the network.
"""
sparsity(net::PEPSNetwork{T,S}) where {T,S} = S

"""
$(TYPEDSIGNATURES)

MpsContractor is a mutable struct that represents the contractor responsible for contracting a PEPS (Projected Entangled Pair States) network using the MPO-MPS (Matrix Product Operator - Matrix Product State) scheme.

# Type Parameters
- `T<:AbstractStrategy`: Specifies the contraction strategy to be employed.
- `R<:AbstractGauge`: Specifies the gauge-fixing method used for optimizing the contraction.
- `S<:Real`: Represents the numeric precision type for real values (e.g., Float64).

# Constructor
This constructor initializes an instance of MpsContractor with the following arguments:
Positional arguments:
- `net`: The PEPS network to be contracted.
- `params`: Contains the control parameters for the MPO-MPS contraction, such as bond dimension and the number of sweeps.
Keyword arguments:
- `beta::S`: The inverse temperature, β, which is crucial for focusing on low-energy states. A larger β sharpens the focus on these states but may reduce the numerical stability of the tensor contraction. The optimal value of β often depends on the problem instance.
- `graduate_truncation::Bool`: A flag indicating whether bond dimensions in the MPS are truncated progressively. When set to true, this truncation method adjusts the bond dimensions gradually during contraction.
- `onGPU::Bool`: A flag indicating whether the computation should be performed on a GPU (default is true).
- `depth::Int`: Specifies the depth of variational sweeps during the Zipper algorithm. A value of 0 implies a full variational sweep across all lattice sites.
The constructor sets up the internal structure of the contractor, including the MPO layers, search order for nodes, and storage for contraction statistics.
"""
mutable struct MpsContractor{T<:AbstractStrategy,R<:AbstractGauge,S<:Real} <:
               AbstractContractor
    peps::PEPSNetwork{T} where {T}
    beta::S
    graduate_truncation::Bool
    depth::Int
    params::MpsParameters{S}
    layers::MpoLayers
    statistics::Any#::Dict{Vector{Int}, <:Real}
    nodes_search_order::Vector{Node}
    node_outside::Node
    node_search_index::Dict{Node,Int}
    current_node::Node
    onGPU::Bool

    function MpsContractor{T,R,S}(
        net,
        params;
        beta::S,
        graduate_truncation::Bool,
        onGPU = true,
        depth::Int = 0,
    ) where {T,R,S}
        ml = MpoLayers(layout(net), net.ncols)
        stat = Dict()
        ord, node_out = nodes_search_order_Mps(net)
        enum_ord = Dict(node => i for (i, node) ∈ enumerate(ord))
        node = ord[begin]
        new(
            net,
            beta,
            graduate_truncation,
            depth,
            params,
            ml,
            stat,
            ord,
            node_out,
            enum_ord,
            node,
            onGPU,
        )
    end
end

function MpsContractor(
    ::Type{T},
    ::Type{R},
    ::Type{S},
    net,
    params;
    beta::S,
    graduate_truncation::Bool,
    onGPU = true,
    depth::Int = 0,
) where {T,R,S}
    return MpsContractor{T,R,S}(net, params; beta, graduate_truncation, onGPU, depth)
end

function MpsContractor(
    ::Type{T},
    ::Type{R},
    net,
    params;
    beta::S,
    graduate_truncation::Bool,
    onGPU = true,
    depth::Int = 0,
) where {T,R,S}
    return MpsContractor(T, R, S, net, params; beta, graduate_truncation, onGPU, depth)
end

function MpsContractor(
    ::Type{T},
    net,
    params;
    beta::S,
    graduate_truncation::Bool,
    onGPU = true,
    depth::Int = 0,
) where {T,S}
    return MpsContractor(T, NoUpdate, net, params; beta, graduate_truncation, onGPU, depth)
end

"""
$(TYPEDSIGNATURES)
Get the strategy used to contract the PEPS network.

# Arguments
- `::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.

# Returns
- `T`: The strategy used for network contraction.
"""
strategy(::MpsContractor{T}) where {T} = T

"""
$(TYPEDSIGNATURES)
Construct and memoize a Matrix Product Operator (MPO) for a given set of layers.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `layers::Dict{Site, Sites}`: A dictionary mapping sites to their corresponding layers.
- `r::Int`: The current row index.

# Returns
- `QMpo`: The constructed MPO for the specified layers.

This function constructs an MPO by iterating through the specified layers and assembling the corresponding tensors. The resulting MPO is memoized for efficient reuse.
"""
@memoize Dict function mpo(
    ctr::MpsContractor{T,R,S},
    layers::Dict{Site,Sites},
    r::Int,
) where {T<:AbstractStrategy,R,S}
    mpo = Dict{Site,MpoTensor{S}}()
    for (site, coordinates) ∈ layers
        lmpo = TensorMap{S}()
        for dr ∈ coordinates
            ten = tensor(ctr.peps, PEPSNode(r + dr, site), ctr.beta)
            push!(lmpo, dr => ten)
        end
        push!(mpo, site => MpoTensor(lmpo))
    end
    ctr.onGPU ? move_to_CUDA!(QMpo(mpo)) : QMpo(mpo)
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the top Matrix Product State (MPS) using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed top MPS for the specified row.

This function constructs the top MPS using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, truncation, and compression steps as needed based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps_top(ctr::MpsContractor{SVDTruncate,R,S}, i::Int) where {R,S}
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i < 1
        W = mpo(ctr, ctr.layers.main, 1)
        return IdentityQMps(S, local_dims(W, :up); onGPU = ctr.onGPU)
    end

    ψ = mps_top(ctr, i - 1)
    W = transpose(mpo(ctr, ctr.layers.main, i))
    ψ0 = dot(W, ψ)

    canonise!(ψ0, :right)
    if ctr.graduate_truncation
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed (bottom) MPS for the specified row.

This function constructs the (bottom) MPS using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, truncation, and compression steps as needed based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps(ctr::MpsContractor{SVDTruncate,R,S}, i::Int) where {R,S}
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows)
        return IdentityQMps(S, local_dims(W, :down); onGPU = ctr.onGPU)
    end

    ψ = mps(ctr, i + 1)


    W = mpo(ctr, ctr.layers.main, i)

    ψ0 = dot(W, ψ)


    canonise!(ψ0, :right)


    if ctr.graduate_truncation
        canonise_truncate!(ψ0, :left, Dcut * 2, tolS / 2)
        variational_sweep!(ψ0, W, ψ, Val(:right))
    end
    canonise_truncate!(ψ0, :left, Dcut, tolS)
    #variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end


"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) approximation using Singular Value Decomposition (SVD) for a given row.

# Arguments
- `ctr::MpsContractor{SVDTruncate}`: The MpsContractor object representing the PEPS network contraction with SVD truncation.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed (bottom) MPS approximation for the specified row.

This function constructs the (bottom) MPS approximation using SVD for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS approximation is memoized for efficient reuse.
"""
@memoize Dict function mps_approx(ctr::MpsContractor{SVDTruncate,R,S}, i::Int) where {R,S}
    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows)
        return IdentityQMps(S, local_dims(W, :down); onGPU = ctr.onGPU) # F64 for now
    end

    W = mpo(ctr, ctr.layers.main, i)
    ψ = IdentityQMps(S, local_dims(W, :down); onGPU = ctr.onGPU) # F64 for now

    ψ0 = dot(W, ψ)
    truncate!(ψ0, :left, ctr.params.bond_dimension)
    ψ0
end

#=
function (ctr::MpsContractor)(peps::PEPSNetwork, ...., :mps_top)

for ctr ∈ [ctr_1, ctr_2]
    mpo_1 = ctr_1(peps, ..., :mpo)
    mpo_2 = ctr_2(peps, ..., :mpo)

    #@nexprs 2 k k -> mpo_k = ctr_k(peps, ..., :mpo)
end
=#

"""
$(TYPEDSIGNATURES)

Construct and memoize the top Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row.

# Arguments
- `ctr::MpsContractor{Zipper}`: The MpsContractor object representing the PEPS network contraction with the Zipper method.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed top MPS using the Zipper method for the specified row.

This function constructs the top Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps_top(ctr::MpsContractor{Zipper,R,S}, i::Int) where {R,S}
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps
    iters_svd = ctr.params.iters_svd
    iters_var = ctr.params.iters_var
    Dtemp_multiplier = ctr.params.Dtemp_multiplier
    method = ctr.params.method
    depth = ctr.depth
    if i < 1
        W = mpo(ctr, ctr.layers.main, 1)
        return IdentityQMps(S, local_dims(W, :up); onGPU = ctr.onGPU) # F64 for now
    end

    ψ = mps_top(ctr, i - 1)
    W = transpose(mpo(ctr, ctr.layers.main, i))

    canonise!(ψ, :left)
    ψ0 = zipper(
        W,
        ψ;
        method = method,
        Dcut = Dcut,
        tol = tolS,
        iters_svd = iters_svd,
        iters_var = iters_var,
        Dtemp_multiplier = Dtemp_multiplier,
        depth = depth,
    )
    canonise!(ψ0, :left)
    variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct and memoize the (bottom) Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row.

# Arguments
- `ctr::MpsContractor{Zipper}`: The MpsContractor object representing the PEPS network contraction with the Zipper method.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed (bottom) MPS using the Zipper method for the specified row.

This function constructs the (bottom) Matrix Product State (MPS) using the Zipper (truncated Singular Value Decomposition) method for a given row in the PEPS network contraction. It recursively builds the MPS row by row, performing canonicalization, and truncation steps based on the specified parameters in `ctr.params`. The resulting MPS is memoized for efficient reuse.
"""
@memoize Dict function mps(ctr::MpsContractor{Zipper,R,S}, i::Int) where {R,S}
    Dcut = ctr.params.bond_dimension
    tolV = ctr.params.variational_tol
    tolS = ctr.params.tol_SVD
    max_sweeps = ctr.params.max_num_sweeps
    iters_svd = ctr.params.iters_svd
    iters_var = ctr.params.iters_var
    Dtemp_multiplier = ctr.params.Dtemp_multiplier
    method = ctr.params.method
    depth = ctr.depth

    if i > ctr.peps.nrows
        W = mpo(ctr, ctr.layers.main, ctr.peps.nrows)
        ψ0 = IdentityQMps(S, local_dims(W, :down); onGPU = ctr.onGPU)
    else
        ψ = mps(ctr, i + 1)
        W = mpo(ctr, ctr.layers.main, i)
        canonise!(ψ, :left)
        ψ0 = zipper(
            W,
            ψ;
            method = method,
            Dcut = Dcut,
            tol = tolS,
            iters_svd = iters_svd,
            iters_var = iters_var,
            Dtemp_multiplier = Dtemp_multiplier,
            depth = depth,
        )
        canonise!(ψ0, :left)
        #variational_compress!(ψ0, W, ψ, tolV, max_sweeps)
    end
    ψ0
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) dressed Matrix Product State (MPS) for a given row and strategy.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.

# Returns
- `QMps`: The constructed dressed MPS for the specified row and strategy.

This function constructs the dressed Matrix Product State (MPS) for a given row in the PEPS network contraction using the specified strategy and memoizes the result for future use. It internally calls other functions such as `mps` and `mpo` to construct the dressed MPS. Additionally, it normalizes the MPS tensors to ensure numerical stability.

Note: The memoization ensures that the dressed MPS is only constructed once for each combination of arguments and is reused when needed.
"""
@memoize Dict function dressed_mps(
    ctr::MpsContractor{T},
    i::Int,
) where {T<:AbstractStrategy}

    ψ = mps(ctr, i + 1)

    caches = Memoization.find_caches(mps)
    delete!(caches[mps], ((ctr, i + 1), ()))
    if ctr.onGPU
        ψ = move_to_CUDA!(ψ)
    end
    W = mpo(ctr, ctr.layers.dress, i)
    ϕ = dot(W, ψ)

    for j ∈ ϕ.sites
        nrm = maximum(abs.(ϕ[j]))
        if !iszero(nrm)
            ϕ[j] ./= nrm
        end
    end
    ϕ
end

"""
$(TYPEDSIGNATURES)

Construct (and memoize) the right environment tensor for a given node in the PEPS network contraction.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.
- `∂v::Vector{Int}`: A vector representing the partial environment configuration.

# Returns
- `Array{S,2}`: The constructed right environment tensor for the specified node.

This function constructs the right environment tensor for a given node in the PEPS network contraction using the specified strategy and memoizes the result for future use. It internally calls other functions such as `dressed_mps` and `mpo` to construct the right environment tensor. Additionally, it normalizes the right environment tensor to ensure numerical stability.

Note: The memoization ensures that the right environment tensor is only constructed once for each combination of arguments and is reused when needed.
"""
@memoize Dict function right_env(
    ctr::MpsContractor{T,R,S},
    i::Int,
    ∂v::Vector{Int},
) where {T<:AbstractStrategy,R,S}
    l = length(∂v)
    if l == 0
        return ctr.onGPU ? CUDA.ones(S, 1, 1) : ones(S, 1, 1)
    end

    R̃ = right_env(ctr, i, ∂v[2:l])
    if ctr.onGPU
        R̃ = CuArray(R̃)
    end
    ϕ = dressed_mps(ctr, i)
    W = mpo(ctr, ctr.layers.right, i)
    k = length(ϕ.sites)
    site = ϕ.sites[k-l+1]
    M = W[site]
    B = ϕ[site]

    RR = update_reduced_env_right(R̃, ∂v[1], M, B)

    ls_mps = left_nbrs_site(site, ϕ.sites)
    ls = left_nbrs_site(site, W.sites)

    while ls > ls_mps
        RR = update_reduced_env_right(RR, W[ls].ctr)
        ls = left_nbrs_site(ls, W.sites)
    end
    nmr = maximum(abs.(RR))
    if ~iszero(nmr)
        RR ./= nmr
    end
    if typeof(RR) <: CuArray
        RR = Array(RR)
    end
    RR
end


"""
$(TYPEDSIGNATURES)
Construct (and memoize) the left environment tensor for a given node in the PEPS network contraction.

# Arguments
- `ctr::MpsContractor{T}`: The MpsContractor object representing the PEPS network contraction.
- `i::Int`: The current row index.
- `∂v::Vector{Int}`: A vector representing the partial environment configuration.

# Returns
- `Array{S,2}`: The constructed left environment tensor for the specified node.

This function constructs the left environment tensor for a given node in the PEPS network contraction using the specified strategy and memoizes the result for future use. It internally calls other functions such as `dressed_mps` to construct the left environment tensor. Additionally, it normalizes the left environment tensor to ensure numerical stability.

Note: The memoization ensures that the left environment tensor is only constructed once for each combination of arguments and is reused when needed.

"""
@memoize Dict function left_env(
    ctr::MpsContractor{T,R,S},
    i::Int,
    ∂v::Vector{Int},
) where {T,R,S}
    l = length(∂v)
    if l == 0
        return ctr.onGPU ? CUDA.ones(S, 1) : ones(S, 1)
    end

    L̃ = left_env(ctr, i, ∂v[1:l-1])
    ϕ = dressed_mps(ctr, i)
    m = ∂v[l]
    site = ϕ.sites[l]
    M = ϕ[site]

    # @matmul L[x] := sum(α) L̃[α] * M[α, x, $m]
    @tensor L[x] := L̃[α] * view(M, :, :, m)[α, x]
    nmr = maximum(abs.(L))
    iszero(nmr) ? L : L ./ nmr
end

"""
$(TYPEDSIGNATURES)
Clear all memoization caches used by the PEPS network contraction.

This function clears all memoization caches that store previously computed results for various operations and environments in the PEPS network contraction.
Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function removes all cached results, which can be useful when you want to free up memory or ensure that the caches are refreshed with updated data.
"""
function clear_memoize_cache()
    Memoization.empty_all_caches!()
end

"""
$(TYPEDSIGNATURES)

Clear memoization caches for specific operations after processing a row.
This function clears the memoization caches for specific operations used in the PEPS network contraction after processing a row.
The cleared operations include `left_env`, `right_env`, `mpo`, and `dressed_mps`. Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function allows you to clear the caches for these specific operations, which can be useful when you want to free up memory or ensure that the caches are refreshed with updated data after processing a row in the contraction.
"""
function clear_memoize_cache_after_row()
    Memoization.empty_cache!.((left_env, right_env, mpo, dressed_mps))
end

"""
$(TYPEDSIGNATURES)
Clear memoization cache for specific operations for a given row and index beta.

This function clears the memoization cache for specific operations used in the PEPS network contraction for a given row.
The cleared operations include `mps_top`, `mps`, `mpo`, `dressed_mps`, and related operations.
Memoization is used to optimize the contraction process by avoiding redundant computations.
Calling this function allows you to clear the cache for these specific operations for a particular row and index beta, which can be useful when you want to free up memory or ensure that the cache is refreshed with updated data for a specific computation.

# Arguments
- `ctr::MpsContractor{T, S}`: The PEPS network contractor object.
- `row::Site`: The row for which the cache should be cleared.

"""
function clear_memoize_cache(ctr::MpsContractor{T,S}, row::Site) where {T,S}
    for i ∈ row:ctr.peps.nrows
        delete!(Memoization.caches[mps_top], ((ctr, i), ()))
    end
    for i ∈ 1:row+1
        delete!(Memoization.caches[mps], ((ctr, i), ()))
    end
    for i ∈ row:row+2
        cmpo = Memoization.caches[mpo]
        delete!(cmpo, ((ctr, ctr.layers.main, i), ()))
        delete!(cmpo, ((ctr, ctr.layers.dress, i), ()))
        delete!(cmpo, ((ctr, ctr.layers.right, i), ()))
    end

end


function sweep_gauges!(
    ctr::MpsContractor{T,GaugeStrategy},
    row::Site,
    tol::Real = 1E-4,
    max_sweeps::Int = 10,
) where {T}
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row)
    ψ_bot = mps(ctr, row + 1)

    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)

    onGPU = ψ_top.onGPU && ψ_bot.onGPU

    gauges = optimize_gauges_for_overlaps!!(ψ_top, ψ_bot, tol, max_sweeps)

    for i ∈ ψ_top.sites
        g = gauges[i]
        g_inv = 1.0 ./ g
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        top = ctr.peps.gauges.data[n_top]
        bot = ctr.peps.gauges.data[n_bot]
        onGPU ? top = CuArray(top) : top
        onGPU ? bot = CuArray(bot) : bot
        g_top = top .* g
        g_bot = bot .* g_inv
        push!(ctr.peps.gauges.data, n_top => g_top, n_bot => g_bot)
    end
    clear_memoize_cache(ctr, row)
end


function sweep_gauges!(
    ctr::MpsContractor{T,GaugeStrategyWithBalancing},
    row::Site,
) where {T}
    clm = ctr.layers.main
    ψ_top = mps_top(ctr, row)
    ψ_bot = mps(ctr, row + 1)
    ψ_top = deepcopy(ψ_top)
    ψ_bot = deepcopy(ψ_bot)
    for i ∈ ψ_top.sites
        @inbounds n_bot = PEPSNode(row + 1 + clm[i][begin], i)
        @inbounds n_top = PEPSNode(row + clm[i][end], i)
        ρ = overlap_density_matrix(ψ_top, ψ_bot, i)
        _, _, scale = LinearAlgebra.LAPACK.gebal!('S', ρ)
        push!(ctr.peps.gauges.data, n_top => 1.0 ./ scale, n_bot => scale)
    end
    clear_memoize_cache(ctr, row)
    ψ_top * ψ_bot
end


function sweep_gauges!(
    ctr::MpsContractor{T,NoUpdate},
    row::Site,
    tol::Real = 1E-4,
    max_sweeps::Int = 10,
) where {T}

end


function update_gauges!(ctr::MpsContractor{T,S}, row::Site, ::Val{:down}) where {T,S}
    for i ∈ 1:row-1
        sweep_gauges!(ctr, i)
    end
end


function update_gauges!(ctr::MpsContractor{T,S}, row::Site, ::Val{:up}) where {T,S}
    for i ∈ row-1:-1:1
        sweep_gauges!(ctr, i)
    end
end

function boundary_states(
    ctr::MpsContractor{T},
    states::Vector{Vector{Int}},
    node::S,
) where {T,S}
    boundary_recipe = boundary(ctr, node)
    res = ones(Int, length(states), length(boundary_recipe))
    for (i, node) ∈ enumerate(boundary_recipe)
        @inbounds res[:, i] = boundary_indices(ctr, node, states)
    end
    [res[r, :] for r ∈ 1:size(res, 1)]
end


function boundary(ctr::MpsContractor{T}, node::Node) where {T}
    boundary(layout(ctr.peps), ctr, node)
end


function local_state_for_node(ctr::MpsContractor{T}, σ::Vector{Int}, w::S) where {T,S}
    k = get(ctr.node_search_index, w, 0)
    0 < k <= length(σ) ? σ[k] : 1
end


function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Union{NTuple{2,S},Tuple{S,NTuple{N,S}}},
    states::Vector{Vector{Int}},
) where {T,S,N}
    v, w = nodes
    if ctr.peps.vertex_map(v) ∈ vertices(ctr.peps.potts_hamiltonian)
        @inbounds idx = [σ[ctr.node_search_index[v]] for σ ∈ states]
        return @inbounds projector(ctr.peps, v, w)[idx]
    end
    ones(Int, length(states))
end

"""
$(TYPEDSIGNATURES)

boundary index formed from outer product of two projectors
"""
function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Union{NTuple{4,S},Tuple{S,NTuple{2,S},S,NTuple{2,S}}},
    states::Vector{Vector{Int}},
) where {S,T}
    v, w, k, l = nodes
    pv = projector(ctr.peps, v, w)
    i = boundary_indices(ctr, (v, w), states)
    j = boundary_indices(ctr, (k, l), states)
    (j .- 1) .* maximum(pv) .+ i
end

function boundary_indices(
    ctr::MpsContractor{T},
    nodes::Tuple{S,NTuple{2,S},S,NTuple{2,S},S,NTuple{2,S},S,NTuple{2,S}},
    states::Vector{Vector{Int}},
) where {S,T}
    v1, v2, v3, v4, v5, v6, v7, v8 = nodes
    pv1 = projector(ctr.peps, v1, v2)
    pv3 = projector(ctr.peps, v3, v4)
    mm = maximum(pv1) * maximum(pv3)
    i = boundary_indices(ctr, (v1, v2, v3, v4), states)
    j = boundary_indices(ctr, (v5, v6, v7, v8), states)
    (j .- 1) .* mm .+ i
end
