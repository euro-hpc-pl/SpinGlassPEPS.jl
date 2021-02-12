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

my_tests = []
if CUDA.functional() && CUDA.has_cutensor() && false
    push!(my_tests,
    "cuda/base.jl",
    "cuda/contractions.jl",
    "cuda/compressions.jl"
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
    "indexing.jl",
    "searchMPS.jl",
    "MPS_search.jl",
    "factor.jl",
    "PEPS.jl",
    #"testing_probabilities_short.jl", # to be removed
    #"testing_probabilities.jl", # to be removed
    "contract.jl",
    "indexing.jl",
    #"notation_tests.jl", # to be removed
    #"peps_tests.jl" # to be removed
)

for my_test in my_tests
    include(my_test)
end
