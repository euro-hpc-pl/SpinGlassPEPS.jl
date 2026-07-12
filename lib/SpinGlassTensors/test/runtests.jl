using SpinGlassTensors
using TensorOperations
using Logging
using LinearAlgebra
using CUDA

disable_logging(LogLevel(1))

using Test

my_tests = ["canonise.jl", "variational.jl", "projectors.jl"]

for my_test in my_tests
    include(my_test)
end
