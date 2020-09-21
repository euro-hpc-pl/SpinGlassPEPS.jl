using Test
using LinearAlgebra
using TensorOperations

path = "/home/kdomino/Dokumenty/julia_modules/"
include(path*"MPStates.jl/src/MPStates.jl")
include(path*"MPStates.jl/src/mps.jl")
include(path*"MPStates.jl/src/mps_operations.jl")
include(path*"MPStates.jl/src/cache.jl")
include(path*"MPStates.jl/src/tensor_contractions.jl")
include(path*"MPStates.jl/src/tensor_factorizations.jl")

include("notation.jl")
include("tests/notation_tests.jl")

include("peps.jl")
include("tests/peps_tests.jl")

#include("arbitrary_peps.jl")
#include("tests/arbitrary_peps_tests.jl")

#include("alternative_graphical.jl")
#include("tests/alternative_graphical_tests.jl")
