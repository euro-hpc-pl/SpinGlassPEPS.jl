using SpinGlassPEPS

using LinearAlgebra
using CUDA
# CUDA.allowscalar(false)
using Test

my_tests = ["MPS.jl", "MPO.jl", "PEPS.jl"]
for my_test in my_tests
    include(my_test)
end