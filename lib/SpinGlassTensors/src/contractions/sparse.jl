#TODO add support for CuSparseMatrixCSR (cf. https://github.com/JuliaGPU/CUDA.jl/issues/1113)

# TODO This function is a patch and may not provide any advantage - to be tested
#=
function CUDA.:*(Md::DenseCuMatrix{T}, Mcsr::CUSPARSE.CuSparseMatrixCSR{T}) where T
    ret = CUDA.zeros(T, size(Md, 1), size(Mcsr, 2))
    CUSPARSE.mm!('T', 'T', one(T), Mcsr, Md, zero(T), ret, 'O')
    ret'
end
=#
#
# TODO shouldn't we have CSR format instead?
function SparseArrays.sparse(::Type{R}, p::CuArray{Int64,1}; mp = nothing) where {R<:Real}
    n = length(p)
    if isnothing(mp)
        mp = maximum(p)
    end
    cn = CuArray(1:n+1)  # aux_cusparse(R, n)
    co = CUDA.ones(R, n)
    CuSparseMatrixCSR(CuSparseMatrixCSC(cn, p, co, (mp, n))) # TODO: Change when CUDA.jl is fixed
end

function SparseArrays.sparse(::Type{R}, p::Vector{Int64}; mp = nothing) where {R<:Real}
    n = length(p)
    if isnothing(mp)
        mp = maximum(p)
    end
    cn = collect(1:n)
    co = ones(R, n)
    sparse(p, cn, co, mp, n)
end

@memoize Dict function SparseArrays.sparse(
    ::Type{T},
    lp::PoolOfProjectors,
    k1::R,
    k2::R,
    k3::R,
    device::Symbol,
) where {T<:Real,R<:Int}
    p1 = get_projector!(lp, k1) #, device)
    p2 = get_projector!(lp, k2) #, device)
    p3 = get_projector!(lp, k3) #, device)
    @assert length(p1) == length(p2) == length(p3)
    s1, s2, s3 = size(lp, k1), size(lp, k2), size(lp, k3)
    p = p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
    if device == :GPU
        p = CuArray(p)
    end
    sparse(T, p; mp = s1 * s2 * s3)
end

@memoize Dict function SparseArrays.sparse(
    ::Type{R},
    lp::PoolOfProjectors,
    k::Int,
    device::Symbol;
    from::Int = 1,
    to::Int = length(lp, k),
) where {R<:Real}
    p = get_projector!(lp, k)
    pp = @view p[from:to]
    rf = minimum(pp)
    rt = maximum(pp)
    if device == :GPU
        pp = CuArray(pp)
    end
    ipr = sparse(R, pp .- (rf - 1))
    (ipr, rf, rt)
end
