using LinearAlgebra
using LightGraphs
using NPZ
using SpinGlassPEPS
using Logging
using ArgParse
using CSV
using Test

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
    "--beta", "-b"
    default = 2.
    arg_type = Float64
    help = "beta value"
    "--chi", "-c"
    default = typemax(Int)
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
end

fi = parse_args(s)["file"]
file = split(fi, "/")[end]
folder = fi[1:end-length(file)]
println(file)
println(folder)

ig = ising_graph(fi)

β = parse_args(s)["beta"]
χ = parse_args(s)["chi"]
spectrum_cutoff = parse_args(s)["spectrum_cutoff"]
node_size = (parse_args(s)["node_size1"], parse_args(s)["node_size2"])
s1 = isqrt(div( nv(ig) ,8))

n = div(s1, node_size[1])
m = div(s1, node_size[2])


D = Dict{Int, Int}()
for v in vertices(ig)
  push!(D, (v => spectrum_cutoff))
end

fg = factor_graph(
      ig,
      D,
      spectrum=brute_force,
      cluster_assignment_rule=super_square_lattice((m, node_size[1], n, node_size[2], 8))
  )


  control_params = Dict(
       "bond_dim" => χ,
       "var_tol" => 1E-12,
       "sweeps" => 4.
   )

peps = PEPSNetwork(m, n, fg, β, :NW, control_params)

n_sols = parse_args(s)["n_sols"]

println(node_size)

@time sols = low_energy_spectrum(peps, n_sols)


energies = sols.energies
println("energies from peps")
for energy in energies
    println(energy)
end

fil = folder*"groundstates_otn2d.txt"
data = split.(readlines(open(fil)))

i = findall(x->x[1]==file, data)[1]
ground_ref = [parse(Int, el) for el in data[i][4:end]]

f(i) = (i == 0)  ? -1 : 1
ground_spins = map(f, ground_ref)
energy_ref = energy(ground_spins, ig)

println("reference energy form real ground state = ", energy_ref)
println("reference energy form file = ", data[i][3])

spins_mat = vecvec2matrix(sols.states)

d = Dict("energies" => energies, "energy_ref" => energy_ref, "chi" => χ, "beta" => β)
npzwrite(folder*file[1:3]*"output.npz", d)
