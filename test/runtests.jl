using SpinGlassPEPS

using LinearAlgebra
using CUDA
# scalar indexing is fine before 0.2
# CUDA.allowscalar(false)

using Test

my_tests = ["MPS.jl", "MPO.jl", "PEPS.jl"]
for my_test in my_tests
    include(my_test)
end
if CUDA.functional()
    include("cuda.jl")
end