module SpinGlassExhaustiveTests

using SpinGlassPEPS.SpinGlassExhaustive
using SpinGlassPEPS.SpinGlassNetworks
using Test
using CUDA
my_tests = ["utils.jl", "brute_force.jl"]

if CUDA.functional()
    pushfirst!(my_tests, "ising.jl")
else
    @info "CUDA is not functional; skipping GPU exhaustive-search tests"
end

for my_test ∈ my_tests
    include(my_test)
end

end # module SpinGlassExhaustiveTests
