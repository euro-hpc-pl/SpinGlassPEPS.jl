using CUDA
using SpinGlassPEPS
using LinearAlgebra
using TensorOperations

using Test

my_tests = []
if CUDA.functional() && CUDA.has_cutensor()
    push!(my_tests,
    # "cuda/base.jl",
    # "cuda/contractions.jl",
    "cuda/compressions.jl"
    )
end

push!(my_tests,
    # "base.jl",
    # "contractions.jl",
    # "compressions.jl",
    # "ising.jl"
)
for my_test in my_tests
    include(my_test)
end
