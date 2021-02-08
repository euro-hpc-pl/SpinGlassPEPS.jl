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
    #"base.jl",
    #"utils.jl",
    #"contractions.jl",
    #"compressions.jl",
    #"ising.jl",
    #"indexing.jl",
    #"searchMPS.jl",
    #"spectrum.jl",
    #"graph.jl",
    #"PEPS.jl",
    "testing_probabilities_short.jl", # NO
    "testing_probabilities.jl", # NO
    "contract.jl",
    #"indexing.jl",
    #"notation_tests.jl",
    "peps_tests.jl", # NO
    "mps_tests.jl", # NO
    "tests_full_graph.jl",
    "tests_on_data.jl"
)

for my_test in my_tests
    include(my_test)
end
