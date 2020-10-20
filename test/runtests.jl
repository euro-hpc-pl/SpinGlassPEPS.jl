using SpinGlassPEPS

using LinearAlgebra
using CUDA
# scalar indexing is fine before 0.2
# CUDA.allowscalar(false)

using Test

if CUDA.functional() && CUDA.has_cutensor()
    include("cuda.jl")
end

my_tests = ["MPS.jl", "MPO.jl", "contractions.jl", "compressions.jl"]
for my_test in my_tests
    include(my_test)
end
