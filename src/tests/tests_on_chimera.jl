using NPZ
using ArgParse
#addprocs(1)

include("../notation.jl")
include("../brute_force.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")

problem_size = 128
file = "001.txt"
β = 2.
χ = 20

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

# reading data from txt
data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(fi))[2:end]))


function make_interactions(data::Array{Array{Any,1},1})
    interactions = Interaction{Float64}[]
    for k in 1:size(data,1)
        i = Int(data[k][1])
        j = Int(data[k][2])
        J = -1*Float64(data[k][3])
        push!(interactions, Interaction((i,j), J))
    end
    interactions
end

# TODO this seams not to work proprtly
function M2Qubo(M::Matrix{T}) where T <: AbstractFloat
    h = diagm(diag(M))
    J = M-h
    a = 4*J
    b = 2*h - 2*diagm(dropdims(sum(J; dims=1); dims=1))
    a+b
end

interactions = make_interactions(data)

M = -1*interactions2M(interactions)

n = Int(sqrt(problem_size/8))

grid, nodes_numbers = form_a_chimera_grid(n)

ns = [Node_of_grid(i,nodes_numbers, grid, interactions; chimera = true) for i in 1:(n*n)]

@time spins, objective = solve(interactions, ns, nodes_numbers, 10; β=β, χ = χ, threshold = 1e-8)

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
