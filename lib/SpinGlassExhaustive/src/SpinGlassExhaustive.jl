"""
Main module for `SpinGlassExhaustive.jl` -- a Julia package for brute-force spin-glass problems with CUDA.
"""

module SpinGlassExhaustive
    using SpinGlassNetworks
    # energy(state_code, J) extends the SpinGlassNetworks generic; a separate
    # function here would make the exported name ambiguous for stack users.
    import SpinGlassNetworks: energy
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