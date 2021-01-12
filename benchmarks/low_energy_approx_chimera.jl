using LinearAlgebra
using LightGraphs
using MetaGraphs
using NPZ
using Plots

using SpinGlassPEPS

using Logging
using ArgParse
using CSV
using Test

import SpinGlassPEPS: solve, solve_mps, M2graph, energy, binary2spins, ising_graph

disable_logging(LogLevel(0))
δ = 0.9
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
    help = "maximal number of solutions"
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
    help = "maximal size of the lower spectrum"
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

ig = ising_graph(fi, si, 1)

n_sols = parse_args(s)["n_sols"]
node_size = (parse_args(s)["node_size1"], parse_args(s)["node_size2"])
println(node_size)

fil = folder*"groundstates_otn2d.txt"
data = split.(readlines(open(fil)))

i = findall(x->x[1]==file, data)[1]
ground_ref = [parse(Int, el) for el in data[i][4:end]]
ground_spins = binary2spins(ground_ref)
energy_ref = energy(ground_spins, ig)

spectrum_cutoff = parse_args(s)["spectrum_cutoff"]

ses = collect(spectrum_cutoff:-10:40)
step = 10
n_s = collect(n_sols:-step:1)

delta_e = ones(length(ses), length(n_s))
cut = ones(Int, length(ses), length(n_s))

function proceed()
  j = 1
  for n_sol in n_s
    i = 1
    for s in ses

      @time spins, _ = solve(ig, n_sol; β=β, χ = χ, threshold = 1e-8, node_size = node_size, spectrum_cutoff = s, δ=δ)

      en = minimum([energy(s, ig) for s in spins])

      cut[i,j] = s
      delta_e[i,j] = (en-energy_ref)/abs(energy_ref)
      i = i+1

      println("spectrum cutoff = ", s)
      println("no sols = ", n_sol)
      println("percentage E = ", (en-energy_ref)/abs(energy_ref))
    end
    j = j+1
  end

  d = Dict("percentage_delta_e" => delta_e, "spectrum_size" => cut, "chi" => χ, "beta" => β, "chimera_size" => si, "no_solutions" => n_s)
  npzwrite(folder*"low_energy_sols/"*file[1:3]*"approx$(si).npz", d)

  p = plot(cut[:,1], delta_e[:,1], title = "low spectr. approx., beta = $β, chi = $χ, chimera_s. = $(si)", label = "n_sols = $(n_s[1])", lw = 1.5, left_margin = 10Plots.mm, bottom_margin = 10Plots.mm)
  xlabel!("size of the low energy spectrum")
  ylabel!("percentage delta E")
  for j in 2:size(cut,2)
    plot!(p, cut[:,j], delta_e[:,j], label = "n_sols = $(n_s[j])", lw = 1.5)
  end
  savefig(folder*"low_energy_sols/"*file[1:3]*"approx$(si).pdf")
end

proceed()
