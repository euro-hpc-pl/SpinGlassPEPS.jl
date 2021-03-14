module SpinGlassPEPS

    using Reexport
    @reexport using SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine

    using Requires
    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
            if CUDA.functional() && CUDA.has_cutensor() && false
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
