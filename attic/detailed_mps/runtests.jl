using SpinGlassPEPS
using LightGraphs
using MetaGraphs
using Random
using Test
using TensorOperations
using LinearAlgebra
using Statistics

import SpinGlassPEPS: M2graph, Partial_sol, update_partial_solution, select_best_solutions

include("../../test/test_helpers.jl")

include("mps_implementation.jl")

include("mps_tests.jl")
include("tests_full_graph.jl")
include("tests_on_data.jl")
