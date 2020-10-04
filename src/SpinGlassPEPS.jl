module SpinGlassPEPS
    using LinearAlgebra
    using TensorOperations

    const ⊗ = kron
    export ⊗

    include("MPO.jl")
    include("MPS.jl")
    include("PEPS.jl")

end