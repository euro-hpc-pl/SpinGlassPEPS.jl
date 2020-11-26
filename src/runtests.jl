using LinearAlgebra
using Requires
using TensorOperations, TensorCast
using LowRankApprox
using LightGraphs
using MetaGraphs
using CSV
using CUDA
using LinearAlgebra
using DocStringExtensions
const product = Iterators.product

using Logging
using NPZ
using Test
using Random
using Statistics


disable_logging(LogLevel(0))

include("base.jl")
include("compressions.jl")
include("contractions.jl")
include("ising.jl")
include("graph.jl")
include("PEPS.jl")
include("search.jl")
include("utils.jl")

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
