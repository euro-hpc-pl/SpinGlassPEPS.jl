using NPZ
using ArgParse
using CSV
#addprocs(1)

include("../notation.jl")
include("../brute_force.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")
include("../ising.jl")


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
  end

fi = parse_args(s)["file"]
file = split(fi, "/")[end]
folder = fi[1:end-length(file)]
println(file)
println(folder)

problem_size = parse_args(s)["size"]
β = parse_args(s)["beta"]
χ = parse_args(s)["chi"]
#si = parse_args(s)["size"]

function make_graph(data::Array{Array{Any,1},1})
    L = Int(maximum(maximum(data)))
    M = zeros(L,L)
    println("problem size = ", size(data,1))

    ig = MetaGraph(L, 0.0)

    set_prop!(ig, :description, "The Ising model.")

    for k in 1:size(data,1)
        i = Int(data[k][1])
        j = Int(data[k][2])
        v = Float64(data[k][3])
        if i == j
            set_prop!(ig, i, :h, v) || error("Node $i missing!")
        else
            add_edge!(ig, i, j) &&
            set_prop!(ig, i, j, :J, v) || error("Cannot add Egde ($i, $j)")
        end
        M[i,j] = M[j,i] = v
    end
    ig, M
end


# TODO this does not work
#ig = ising_graph(fi, size)

# reading data from txt
data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(fi))[2:end]))


g, M = make_graph(data)

@time spins, objective = solve(g, 10; β=β, χ = χ, threshold = 1e-8)

energies = [v2energy(M, s) for s in spins]

println("energies from peps")
for i in 1:10
    println(energies[i])
end

fil = folder*"groundstates_otn2d.txt"
data = split.(readlines(open(fil)))

i = findall(x->x[1]==file, data)[1]
ground_ref = [parse(Int, el) for el in data[i][4:end]]
ground_spins = binary2spins(ground_ref)

energy_ref = v2energy(M, ground_spins)
println("reference energy = ", energy_ref)

spins_mat = vecvec2matrix(spins)

d = Dict("spins" => spins_mat, "spins_ref" => ground_spins, "energies" => energies, "energy_ref" => energy_ref, "chi" => χ, "beta" => β)
npzwrite(folder*file[1:3]*"output.npz", d)
