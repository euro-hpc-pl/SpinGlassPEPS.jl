using CUDA
using SpinGlassPEPS
using LinearAlgebra
using TensorOperations

using Test

function reshape_row(A::AbstractArray{T}, dims::Tuple) where {T <: Number}
    ord = reverse(1:length(dims))

    A = reshape(A, reverse(dims))
    permutedims(A, ord)
end 

my_tests = []
if CUDA.functional() && CUDA.has_cutensor() && false
    if length(ARGS) == 0 || "cuda" ∈ ARGS
        push!(my_tests, readdir("cuda", join=true)...)
    end
end

if length(ARGS) == 0 || "unit" ∈ ARGS
    push!(my_tests, readdir("unit", join=true)...)
end

if length(ARGS) == 0 || "integration" ∈ ARGS
    push!(my_tests, readdir("integration", join=true)...)
end
# push!(my_tests,
# #    "base.jl",
# #   "contractions.jl",
# #   "compressions.jl",
# #   "ising.jl",
# #   "search.jl"
#    "graph.jl"
# )


@show my_tests
for my_test in my_tests
    include(my_test)
end
