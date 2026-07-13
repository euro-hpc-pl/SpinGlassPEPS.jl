module SpinGlassEngineTests

using SpinGlassPEPS.SpinGlassEngine
using SpinGlassPEPS.SpinGlassNetworks
using SpinGlassPEPS.SpinGlassTensors
using SpinGlassPEPS.SpinGlassExhaustive
using Logging
using Graphs
using LinearAlgebra
#using Statistics
#using MetaGraphs
using CUDA

disable_logging(LogLevel(1))

user_onGPU = true # or true, based on user's preference
gpu_available = CUDA.functional()
onGPU = user_onGPU && gpu_available

using Test
my_tests = []

push!(
    my_tests,
    #quick tests:
    "api_compat.jl",
    "projector_workspace.jl",
    "geometry_protocol.jl",
    "known_failures.jl",
    "search_squarecross_double_node_basic.jl",
    "operations.jl",
    "branch_and_bound.jl",
    "search_chimera_pathological.jl",
    "search_chimera_smallest.jl",
    "search_cross_square_star.jl",
    "search_smallest_cross_square_cross.jl",
    "search_pegasus_square_cross.jl",
    "search_pegasus_nodiag_square_cross.jl",
    "search_square_double_node_basic.jl",
    "chimera_overlap_python.jl",
    "hamming.jl",
    "search_chimera_smallest_droplets.jl",
    "search_chimera_pathological_droplets.jl",
    "search_chimera_pathological_hamming.jl",
    "search_chimera_pathological_Z2.jl",
    # time consuming tests:
    #    "search_chimera_full.jl",
)

# This is work in progress (may or may not be included in future versions)

push!(
    my_tests,
    # "experimental/zipper.jl",
    # "experimental/truncate.jl",
    # "experimental/truncate_small.jl",
    # "experimental/mpo_size.jl",
    # "experimental/squarestar_double_node_pegasus.jl",
    # "experimental/squarestar_double_node_zephyr.jl",
    # "experimental/gauges_cuda.jl",
    # "experimental/sampling.jl"
    # "experimental/search_chimera_gauge.jl",
    # "experimental/gauges.jl"
)

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

@time begin
    # The droplet searches peak around 10 GiB inside a single solve (plus pool
# reservation): they OOM 3080-class cards and need a quiet >= 12 GiB device.
if !onGPU || CUDA.total_memory() >= 12 * 2^30
    push!(my_tests, "search_chimera_droplets.jl", "search_pegasus_droplets.jl")
else
    @warn "Skipping droplet search tests: GPU has $(Base.format_bytes(CUDA.total_memory())), needs >= 12 GiB"
end

for my_test ∈ my_tests
        include(my_test)
    end
end

end # module SpinGlassEngineTests
