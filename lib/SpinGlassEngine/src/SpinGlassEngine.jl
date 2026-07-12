module SpinGlassEngine

using Base: Tuple
using Base.Cartesian
using CUDA
using SpinGlassTensors
using SpinGlassNetworks
using TensorOperations
using MetaGraphs
using Memoization
using LinearAlgebra, MKL
using Graphs
using ProgressMeter
using Statistics
using DocStringExtensions
using NNlib

include("operations.jl")
include("geometry.jl")
include("PEPS.jl")
include("contractor.jl")
include("square_single_node.jl")
include("king_single_node.jl")
include("square_double_node.jl")
include("square_cross_double_node.jl")
include("tensors.jl")
include("droplets.jl")
include("search.jl")
include("util.jl")

end # module
