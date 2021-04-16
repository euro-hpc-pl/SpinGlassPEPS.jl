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


disable_logging(LogLevel(0))
δH = 0.9
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
    "--lower_cutoff", "-v"
    default = 1
    arg_type = Int
    help = "minimal size of the lower spectrum"
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
lower_cutoff = parse_args(s)["lower_cutoff"]

n_sols = parse_args(s)["n_sols"]
node_size = (parse_args(s)["node_size1"], parse_args(s)["node_size2"])
println(node_size)


s1 = isqrt(div( nv(ig), 8))
n = div(s1, node_size[1])
m = div(s1, node_size[2])


fil = folder*"groundstates_otn2d.txt"
data = split.(readlines(open(fil)))

i = findall(x->x[1]==file, data)[1]
ground_ref = [parse(Int, el) for el in data[i][4:end]]
#TODO this may need to be corrected
f(i) = (i == 0)  ? -1 : 1
ground_spins = map(f, ground_ref)
println(ground_spins)
energy_ref = energy(ground_spins, ig)
println(energy_ref)

ses = spectrum_cutoff:-1:lower_cutoff
step = 10
n_s = n_sols:-step:1

delta_e = ones(length(ses), length(n_s))
cut = ones(Int, length(ses), length(n_s))
rule = super_square_lattice((m, node_size[1], n, node_size[2], 8))

function proceed()
  j = 1
  for n_sol in n_s
    i = 1
    for sc in ses
      println(sc)
      D = Dict{Int, Int}()
      for v in vertices(ig)
        push!(D, (v => sc))
      end
      fg = factor_graph(
            ig,
            D,
            cluster_assignment_rule=rule,
            spectrum=brute_force,
        )

      println("..............")
      for v in vertices(fg)
        println(v, " ", length(props(fg, v)[:spectrum].energies))
      end
      println("..............")

      control_params = Dict(
           "bond_dim" => χ,
           "var_tol" => 1E-12,
           "sweeps" => 4.
       )

      peps = PEPSNetwork(m, n, fg, β, :NW, control_params)

      @time sols = low_energy_spectrum(peps, n_sol)


      en = minimum(sold.energies)

      cut[i,j] = sc
      delta_e[i,j] = (en-energy_ref)/abs(energy_ref)
      i = i+1

      println("spectrum cutoff = ", sc)
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
