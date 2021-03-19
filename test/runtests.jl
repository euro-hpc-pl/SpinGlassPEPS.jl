using Test, CUDA, SpinGlassPEPS

my_tests = []
if CUDA.functional() && CUDA.has_cutensor() && false
    CUDA.allowscalar(false)
    include("cuda/test_helpers.jl")
    push!(
        my_tests,
        "cuda/base.jl",
        "cuda/contractions.jl",
        "cuda/compressions.jl",
        "cuda/spectrum.jl"
    )
end

for my_test in my_tests
    include(my_test)
end