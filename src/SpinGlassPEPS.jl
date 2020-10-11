module SpinGlassPEPS
    using LinearAlgebra
    using Requires
    using TensorOperations


    include("graph.jl")
    include("MPO.jl")
    include("MPS.jl")
    include("PEPS.jl")
    include("utils.jl")

    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
            if CUDA.functional() && CUDA.has_cutensor()
                const CuArray = CUDA.CuArray
                const CuVector = CUDA.CuVector
                const CuMatrix = CUDA.CuMatrix
                include("cuda.jl") 
            end
        end
    end
end