using SpinGlassPEPS

using LinearAlgebra
using CUDA
# scalar indexing is fine before 0.2
# CUDA.allowscalar(false)

using Test

my_tests = ["compressions.jl"] #["compressions.jl"]#["MPS.jl", "MPO.jl", "contractions.jl", "compressions.jl"]
for my_test in my_tests
    include(my_test)
end

if CUDA.functional() && false #skip cuda for now
    include("cuda.jl")
end