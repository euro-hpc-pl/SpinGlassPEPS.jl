using Test
using LinearAlgebra
using TensorOperations
using TensorCast
using Random
using LightGraphs
using MetaGraphs
using LowRankApprox
using Logging

disable_logging(LogLevel(0))

#path = "/home/kdomino/Dokumenty/julia_modules/"
#try
#    include(path*"MPStates.jl/src/MPStates.jl")
#    include(path*"MPStates.jl/src/mps.jl")
#    include(path*"MPStates.jl/src/mps_operations.jl")
#    include(path*"MPStates.jl/src/cache.jl")
#    include(path*"MPStates.jl/src/tensor_contractions.jl")
#    include(path*"MPStates.jl/src/tensor_factorizations.jl")
#catch
#    println("no MPStates.jl in", path)
#end

include("utils.jl")
include("base.jl")
include("contractions.jl")
include("compressions.jl")
include("ising.jl")
include("search.jl")
include("tests/test_helpers.jl")

include("notation.jl")
include("brute_force.jl")
include("tests/notation_tests.jl")

include("peps_no_types.jl")
include("tests/peps_tests.jl")

include("mps_implementation.jl")
include("tests/mps_tests.jl")

include("tests/tests_on_data.jl")
include("tests/tests_of_solvers.jl")

#include("compression.jl")
#include("tests/compression_tests.jl")
