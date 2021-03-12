module SpinGlassPEPS
    using LinearAlgebra
    using Requires
    using TensorOperations, TensorCast
    using LowRankApprox
    using LightGraphs
    using MetaGraphs
    using CSV
    using Logging
    using StatsBase
    using Memoize, LRUCache

    using DocStringExtensions
    const product = Iterators.product

    include("base.jl")
    include("utils.jl")
    include("compressions.jl")
    include("identities.jl")
    include("contractions.jl")
    include("lattice.jl")
    include("ising.jl")
    include("exact.jl")
    include("factor.jl")
    include("search.jl")
    include("PEPS.jl")
    include("MPS_search.jl")

    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
            if CUDA.functional() && CUDA.has_cutensor()
                const CuArray = CUDA.CuArray
                const CuVector = CUDA.CuVector
                const CuMatrix = CUDA.CuMatrix
                const CuSVD = CUDA.CUSOLVER.CuSVD
                const CuQR = CUDA.CUSOLVER.CuQR
                const cu = CUDA.cu
                using CUDA
                # const cuda = CUDA.@cuda

                CUDA.allowscalar(false)

                @inline function cudiv(x::Int)
                    max_threads = 256
                    threads_x = min(max_threads, x)
                    threads_x, ceil(Int, x/threads_x)
                end
                include("cuda/base.jl")
                include("cuda/utils.jl")
                include("cuda/contractions.jl")
                include("cuda/compressions.jl")
                include("cuda/spectrum.jl")
            end
        end
    end
end
