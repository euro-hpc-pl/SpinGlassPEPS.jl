using SpinGlassPEPS

using LinearAlgebra
using CUDA


using Test

if CUDA.functional() && CUDA.has_cutensor() && false
    include("cuda.jl")
end

my_tests = ["Ising.jl"] #["compressions.jl"]#["MPS.jl", "MPO.jl", "contractions.jl", "compressions.jl"]
for my_test in my_tests
    include(my_test)
end
