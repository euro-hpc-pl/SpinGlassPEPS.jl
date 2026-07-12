module SpinGlassNetworks
using LabelledGraphs
using Graphs
using MetaGraphs # TODO: to be replaced by MetaGraphsNext
using CSV
using DocStringExtensions
using LinearAlgebra, MKL
using Base.Cartesian
using SparseArrays
using CUDA, CUDA.CUSPARSE
using SpinGlassTensors
using JLD2
import Base.Prehashed


include("ising.jl")
include("spectrum.jl")
include("lattice.jl")
include("potts_hamiltonian.jl")
include("bp.jl")
include("truncate.jl")
include("utils.jl")
end # module
