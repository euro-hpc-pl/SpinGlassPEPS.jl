# ./mps/identity.jl: This file provides custom MPS Identity. Note, this approach is easier than
#                    trying to overload the universal identity operator, I, from LinearAlgebra.

export local_dims, IdentityQMps

function IdentityQMps(
    ::Type{T},
    loc_dims::Dict,
    Dmax::Int = 1;
    onGPU = true,
) where {T<:Real}
    _zeros = onGPU ? CUDA.zeros : zeros
    id = TensorMap{T}(keys(loc_dims) .=> _zeros.(T, Dmax, Dmax, values(loc_dims)))

    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = _zeros(T, 1, 1, ld_min)
    else
        id[site_min] = _zeros(T, 1, Dmax, ld_min)
        id[site_max] = _zeros(T, Dmax, 1, ld_max)
    end

    for (site, ld) ∈ loc_dims
        id[site][1, 1, :] .= 1 / sqrt(ld)
    end
    QMps(id; onGPU = onGPU)
end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    dim = dir == :down ? 4 : 2
    Dict{Site,Int}(k => size(mpo[k], dim) for k ∈ mpo.sites if ndims(mpo[k]) == 4)
end
