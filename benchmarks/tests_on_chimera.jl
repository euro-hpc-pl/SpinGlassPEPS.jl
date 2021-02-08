using LinearAlgebra
using LightGraphs
using MetaGraphs
using NPZ

using SpinGlassPEPS

using Logging
using ArgParse
using CSV
using Test

import SpinGlassPEPS: solve, solve_mps, M2graph, energy, binary2spins, ising_graph

disable_logging(LogLevel(0))

# this is axiliary function for npz write

function vecvec2matrix(v::Vector{Vector{Int}})
    M = v[1]
    for i in 2:length(v)
        M = hcat(M, v[i])
    end
    M
end

s = ArgParseSettings("description")
  @add_arg_table! s begin
    "--file", "-i"
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
    "--spectrum_cutoff", "-u"
    default = 256
    arg_type = Int
    help = "size of the lower spectrum"
    "--deltaH", "-d"
    default = 0.1
    arg_type = Float64
    help = "merge parameter on merging dX, the threshold on ratios of objectives"
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
δH = parse_args(s)["deltaH"]
spectrum_cutoff = parse_args(s)["spectrum_cutoff"]

node_size = (parse_args(s)["node_size1"], parse_args(s)["node_size2"])
s1 = Int(sqrt(si/8))

n = ceil(Int, s1/node_size[1])
m = ceil(Int, s1/node_size[2])

ig = ising_graph(fi, si, 1)
update_cells!(
    ig,
    rule = square_lattice((m, node_size[1], n, node_size[2], 8)),
  )

fg = factor_graph(
      ig,
      spectrum_cutoff,
      energy=energy,
      spectrum=brute_force,
  )

peps = PepsNetwork(m, n, fg, β, :NW)

n_sols = parse_args(s)["n_sols"]

println(node_size)

@time sols = solve(peps, n_sols; β=β, χ = χ, threshold = 1e-8, node_size = node_size, spectrum_cutoff = spectrum_cutoff, δH=δH)
objective = [e.objective for e in sols]
spins = return_solution(ig, fg, sols)

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
