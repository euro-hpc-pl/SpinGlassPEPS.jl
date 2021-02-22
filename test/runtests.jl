using CUDA
using SpinGlassPEPS
using LinearAlgebra
using TensorOperations
using TensorCast
using LightGraphs
using MetaGraphs
using Random
using Logging
using Statistics
using NPZ

disable_logging(LogLevel(1))

using Test

include("test_helpers.jl")

my_tests = []
if CUDA.functional() && CUDA.has_cutensor()
    CUDA.allowscalar(false)
    include("cuda/test_helpers.jl")
    push!(my_tests,
    "cuda/base.jl",
    "cuda/contractions.jl",
    "cuda/compressions.jl",
    "cuda/spectrum.jl"
    )
end

include("test_helpers.jl")
push!(my_tests,
    "base.jl",
    "utils.jl",
    "contractions.jl",
    "compressions.jl",
    "identities.jl",
    "ising.jl",
    "MPS_search.jl",
    "factor.jl",
    "PEPS.jl",
    "contract.jl",
)

for my_test in my_tests
    include(my_test)
end
