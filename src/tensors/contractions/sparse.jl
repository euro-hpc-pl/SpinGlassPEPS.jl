# Projector indicator matrices: column k carries a single 1 at row p[k].
# These were previously pirated SparseArrays.sparse methods, memoized in a
# process-global cache that SpinGlassEngine had to clear by reaching into
# this package; they now live in the owning PoolOfProjectors.

# TODO shouldn't we have CSR format instead?
function projector_matrix(::Type{R}, p::CuArray{Int64,1}; mp = nothing) where {R<:Real}
    n = length(p)
    if isnothing(mp)
        mp = maximum(p)
    end
    cn = CuArray(1:n+1)
    co = CUDA.ones(R, n)
    CuSparseMatrixCSR(CuSparseMatrixCSC(cn, p, co, (mp, n))) # TODO: Change when CUDA.jl is fixed
end

function projector_matrix(::Type{R}, p::Vector{Int64}; mp = nothing) where {R<:Real}
    n = length(p)
    if isnothing(mp)
        mp = maximum(p)
    end
    cn = collect(1:n)
    co = ones(R, n)
    sparse(p, cn, co, mp, n)
end

function projector_matrix(
    ::Type{T},
    lp::PoolOfProjectors,
    k1::R,
    k2::R,
    k3::R,
    device::Symbol,
) where {T<:Real,R<:Int}
    get!(lp.matrices, (:triple, T, k1, k2, k3, device)) do
        p1 = get_projector!(lp, k1)
        p2 = get_projector!(lp, k2)
        p3 = get_projector!(lp, k3)
        @assert length(p1) == length(p2) == length(p3)
        s1, s2, s3 = size(lp, k1), size(lp, k2), size(lp, k3)
        p = p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
        if device == :GPU
            p = CuArray(p)
        end
        projector_matrix(T, p; mp = s1 * s2 * s3)
    end
end

function projector_matrix(
    ::Type{R},
    lp::PoolOfProjectors,
    k::Int,
    device::Symbol;
    from::Int = 1,
    to::Int = length(lp, k),
) where {R<:Real}
    get!(lp.matrices, (:range, R, k, from, to, device)) do
        p = get_projector!(lp, k)
        pp = @view p[from:to]
        rf = minimum(pp)
        rt = maximum(pp)
        if device == :GPU
            pp = CuArray(pp)
        end
        ipr = projector_matrix(R, pp .- (rf - 1))
        (ipr, rf, rt)
    end
end
