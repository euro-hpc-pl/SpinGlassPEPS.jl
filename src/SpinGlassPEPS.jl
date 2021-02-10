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

    using DocStringExtensions
    const product = Iterators.product

    include("base.jl")
    include("utils.jl")
    include("compressions.jl")
    include("contractions.jl")
    include("lattice.jl")
    #include("graphs/chimera.jl")
    #include("graphs/lattice.jl")
    include("graph.jl")
    include("ising.jl")
    include("PEPS.jl")
    include("spectrum.jl")
    include("notation.jl")
    include("peps_no_types.jl")
    include("mps_implementation.jl")

    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
            if CUDA.functional() && CUDA.has_cutensor()
                const CuArray = CUDA.CuArray
                const CuVector = CUDA.CuVector
                const CuMatrix = CUDA.CuMatrix
                const CuSVD = CUDA.CUSOLVER.CuSVD
                const CuQR = CUDA.CUSOLVER.CuQR
                const cu = CUDA.cu
                CUDA.allowscalar(false)
                include("cuda/base.jl")
                include("cuda/contractions.jl")
                include("cuda/compressions.jl")
            end
        end
    end
end
