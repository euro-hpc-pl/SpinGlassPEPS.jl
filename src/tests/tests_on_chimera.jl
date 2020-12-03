using LinearAlgebra
using Requires
using TensorOperations, TensorCast
using LowRankApprox
using LightGraphs
using MetaGraphs
using CSV
using CUDA
using LinearAlgebra
using DocStringExtensions
const product = Iterators.product

using NPZ
using Logging
using ArgParse
using CSV

disable_logging(LogLevel(0))

include("../base.jl")
include("../compressions.jl")
include("../contractions.jl")
include("../ising.jl")
include("../graph.jl")
include("../PEPS.jl")
include("../search.jl")
include("../utils.jl")

include("../notation.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")



s = ArgParseSettings("description")
  @add_arg_table! s begin
    "--file", "-f"
    arg_type = String
    help = "the file name"
    "--size", "-s"
    default = 128
    arg_type = Int
    help = "problem size"
    "--beta", "-b"
    default = 2.
    arg_type = Float64
    help = "beta value"
    "--chi", "-c"
    default = 20
    arg_type = Int
    help = "cutoff size"
    "--n_sols", "-n"
    default = 10
    arg_type = Int
    help = "number of solutions"
    "--node_size1", "-r"
    default = 1
    arg_type = Int
    help = "chimera node size in rows"
    "--node_size2", "-o"
    default = 1
    arg_type = Int
    help = "chimera node size in columns"
  end

fi = parse_args(s)["file"]
file = split(fi, "/")[end]
folder = fi[1:end-length(file)]
println(file)
println(folder)

problem_size = parse_args(s)["size"]
β = parse_args(s)["beta"]
χ = parse_args(s)["chi"]
si = parse_args(s)["size"]

ig = ising_graph(fi, si, 1, -1)

n_sols = parse_args(s)["n_sols"]
node_size = (parse_args(s)["node_size1"], parse_args(s)["node_size2"])
println(node_size)

@time spins, objective = solve(ig, n_sols; β=β, χ = χ, threshold = 1e-8, node_size = node_size)

energies = [energy(s, ig) for s in spins]
println("energies from peps")
for energy in energies
    println(energy)
end

fil = folder*"groundstates_otn2d.txt"
data = split.(readlines(open(fil)))

i = findall(x->x[1]==file, data)[1]
ground_ref = [parse(Int, el) for el in data[i][4:end]]
ground_spins = binary2spins(ground_ref)
energy_ref = energy(ground_spins, ig)

println("reference energy form data = ", energy_ref)
println("reference energy form file = ", data[i][3])

spins_mat = vecvec2matrix(spins)

d = Dict("spins" => spins_mat, "spins_ref" => ground_spins, "energies" => energies, "energy_ref" => energy_ref, "chi" => χ, "beta" => β)
npzwrite(folder*file[1:3]*"output.npz", d)
