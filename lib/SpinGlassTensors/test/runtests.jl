using SpinGlassTensors
using TensorOperations
using Logging
using LinearAlgebra
using CUDA

disable_logging(LogLevel(1))

using Test

my_tests = [
    "canonise.jl",
    "variational.jl",
    "projectors.jl",
    "linear_algebra_ext.jl",
    "contractions_dense.jl",
    "contractions_site.jl",
    "contractions_virtual.jl",
    "contractions_central.jl",
    "contractions_diagonal.jl",
]

for my_test in my_tests
    include(my_test)
end
