module SpinGlassPEPS
    using LinearAlgebra
    using TensorOperations
    using CUDA

    const ⊗ = kron
    export ⊗

    include("graph.jl")
    include("MPO.jl")
    include("MPS.jl")
    include("PEPS.jl")
end