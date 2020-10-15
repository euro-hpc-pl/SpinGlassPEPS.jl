using NPZ
using Plots
using Test

include("../notation.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")

data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open("data/Qfile_tests.txt"))))

function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)

    transpose(v)*M*v + transpose(v)*d
end

M = zeros(48,48)

q_vec = Qubo_el{Float64}[]
for d in data
    i = ceil(Int, d[1])+1
    j = ceil(Int, d[2])+1
    if d[3] != 0.
        push!(q_vec, Qubo_el((i,j), d[3]))
        M[i,j] = d[3]
    end
end

for i in 1:10
    println(q_vec[i])
end

χ = 15
β = 2.
β_step = 2

print("mps time  =  ")
ns = [Node_of_grid(i, q_vec) for i in 1:get_system_size(q_vec)]
@time spins_mps, objective_mps = solve_mps(q_vec, ns, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-8)

println(spins_mps)
for spin in spins_mps
    println(v2energy(M, spin))
end
