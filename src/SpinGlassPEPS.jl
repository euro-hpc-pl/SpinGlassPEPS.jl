module SpinGlassPEPS
    using LinearAlgebra
    using CUDA
    @show(CUDA.has_cutensor())
    using TensorOperations

    
    include("graph.jl")
    include("MPO.jl")
    include("MPS.jl")
    include("PEPS.jl")
    include("utils.jl")
end