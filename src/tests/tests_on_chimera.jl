using Distributed
#addprocs(1)

include("../notation.jl")
include("../brute_force.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")


folder = "./data/chimera128_spinglass_power/"
file = "002.txt"
file = folder*file
problem_size = 128

head = split.(readlines(open(file))[1:1])
println(head)
# reading data from txt
data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(file))[2:end]))


function make_interactions(data::Array{Array{Any,1},1})
    interactions = Interaction{Float64}[]
    for k in 1:size(data,1)
        i = Int(data[k][1])
        j = Int(data[k][2])
        J = Float64(data[k][3])
        push!(interactions, Interaction((i,j), J))
    end
    interactions
end

interactions = make_interactions(data)

n = Int(sqrt(problem_size/8))
nodes_numbers = nxmgrid(n,n)

grid = Array{Array{Int}}(undef, (n,n))

for i in 1:n
    for j in 1:n
        grid[i,j] = chimera_cell(i,j, problem_size)
    end
end

ns = [Node_of_grid(i,nodes_numbers, grid; chimera = true) for i in 1:(n*n)]

β = 2.
χ = 50

spins, objective = solve(interactions, ns, nodes_numbers, 2; β=β, χ = χ, threshold = 1e-8)

energies = [v2energy(M, s) for s in spins]

println(objective)
println(spins)
println(energies)

npzwrite(folder*file[1:3]*"output.npz", Dict("spins" => spins, "objective" => objective, "energies" => energies))
