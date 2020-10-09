module SpinGlassPEPS
    using LinearAlgebra
    using TensorOperations
    using CUDA

    const ⊗ = kron
    export ⊗

    dg(M::Array{T, 4}) where {T} = permutedims(conj.(M), (2, 1, 3, 4))
    dg(M::Array{T, 3}) where {T} = permutedims(conj.(M), (2, 1, 3))
    
    include("graph.jl")
    include("MPO.jl")
    include("MPS.jl")
    include("PEPS.jl")
end