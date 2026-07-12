"""
Main module for `SpinGlassExhaustive.jl` -- a Julia package for brute-force spin-glass problems with CUDA.
"""

module SpinGlassExhaustive
    using SpinGlassNetworks
    using Graphs
    using LabelledGraphs
    using Bits
    using LinearAlgebra
    using DocStringExtensions
    using CUDA

    include("naive.jl")
    include("utils.jl")
    include("ising.jl")
end