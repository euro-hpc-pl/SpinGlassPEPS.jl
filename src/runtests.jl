using Test
using LinearAlgebra
using TensorOperations
using Random

path = "/home/kdomino/Dokumenty/julia_modules/"
try
    include(path*"MPStates.jl/src/MPStates.jl")
    include(path*"MPStates.jl/src/mps.jl")
    include(path*"MPStates.jl/src/mps_operations.jl")
    include(path*"MPStates.jl/src/cache.jl")
    include(path*"MPStates.jl/src/tensor_contractions.jl")
    include(path*"MPStates.jl/src/tensor_factorizations.jl")
catch
    println("no MPStates.jl in", path)
end

include("notation.jl")
include("compression.jl")
include("peps.jl")
include("mps_implementation.jl")

include("tests/notation_tests.jl")
include("tests/compression_tests.jl")
include("tests/peps_tests.jl")
include("tests/mps_tests.jl")
