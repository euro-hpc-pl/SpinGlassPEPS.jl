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

disable_logging(LogLevel(0))

using Test

function reshape_row(A::AbstractArray{T}, dims::Tuple) where {T <: Number}
    ord = reverse(1:length(dims))

    A = reshape(A, reverse(dims))
    permutedims(A, ord)
end

my_tests = []
if CUDA.functional() && CUDA.has_cutensor()
    push!(my_tests,
    "cuda/base.jl",
    "cuda/contractions.jl",
    "cuda/compressions.jl"
    )
end

push!(my_tests,
    "base.jl",
    "contractions.jl",
    "compressions.jl",
    "ising.jl",
    "search.jl",
    "graph.jl",
    "notation_tests.jl",
   "peps_tests.jl",
   "mps_tests.jl",
   "tests_full_graph.jl",
   "tests_on_data.jl"
)

for my_test in my_tests
    include(my_test)
end
