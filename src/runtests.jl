using Test
using LinearAlgebra
using TensorOperations
using TensorCast
using Random
using Statistics
using LightGraphs
using MetaGraphs
using LowRankApprox
using Logging
using NPZ


disable_logging(LogLevel(0))


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
