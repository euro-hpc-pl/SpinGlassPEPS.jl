
# ./mps/rand.jl: This file provides methods to generate random MPS / MPO

function Base.rand(
    ::Type{QMps{T}},
    loc_dims::Dict,
    Dmax::Int = 1;
    onGPU = false,
) where {T<:Real}
    id = TensorMap{T}(keys(loc_dims) .=> rand.(T, Dmax, Dmax, values(loc_dims)))
    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = rand(T, 1, 1, ld_min)
    else
        id[site_min] = rand(T, 1, Dmax, ld_min)
        id[site_max] = rand(T, Dmax, 1, ld_max)
    end
    onGPU ? move_to_CUDA!(QMps(id)) : QMps(id)
end

function Base.rand(::Type{CentralTensor{T}}, s::Vector{Int}) where {T<:Real}
    CentralTensor(
        Real.(rand(s[1], s[5])),
        Real.(rand(s[2], s[6])),
        Real.(rand(s[3], s[7])),
        Real.(rand(s[4], s[8])),
    )
end

function Base.rand(
    ::Type{SiteTensor{T}},
    lp::PoolOfProjectors,
    l::Int,
    D::NTuple,
) where {T<:Real}
    loc_exp = rand(l)
    projs = D

    SiteTensor(lp, loc_exp, projs)
end

function Base.rand(::Type{QMpo{T}}, loc_dims::Dict; onGPU::Bool = false) where {T<:Real}
    QMpo(MpoTensorMap{T}(loc_dims))
end
