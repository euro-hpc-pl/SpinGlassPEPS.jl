using NPZ
using Plots
using Test
using LinearAlgebra

include("../notation.jl")
include("../compression.jl")
include("../peps_no_types.jl")
include("../mps_implementation.jl")

file = "data/Qfile_tests.txt"
s = 48
file = "data/QUBO8qbits"
s = 8
#file = "data/QUBO6qbits"
# s = 6

data = (x-> Array{Any}([parse(Int, x[1]), parse(Int, x[2]), parse(Float64, x[3])])).(split.(readlines(open(file))))

function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)

    transpose(v)*M*v + transpose(v)*d
end

M = zeros(s,s)


for d in data
    i = ceil(Int, d[1])+1
    j = ceil(Int, d[2])+1
    M[i,j] = d[3]
end

function M2qubo(Q::Matrix{T}) where T <: AbstractFloat
    J = (Q - diagm(diag(Q)))/4
    v = dropdims(sum(Q; dims=1); dims = 1)
    h = diagm(diag(Q)/2 + v/2)
    J + h
end


J = -1*M2qubo(M)

function M2Qubbo_els(M::Matrix{Float64}, T::Type = Float64)
    qubo = Qubo_el{T}[]
    s = size(M)
    for i in 1:s[1]
        for j in i:s[2]
            if (M[i,j] != 0.) | (i == j)
                x = T(M[i,j])
                q = Qubo_el{T}((i,j), x)
                push!(qubo, q)
            end
        end
    end
    qubo
end

q_vec = M2Qubbo_els(J)


χ = 50
χ = 15
β = 1.
β_step = 2

print("mps time  =  ")
ns = [Node_of_grid(i, q_vec) for i in 1:get_system_size(q_vec)]
@time spins_mps, objective_mps = solve_mps(q_vec, ns, 10; β=β, β_step=β_step, χ=χ, threshold = 1.e-12)



#println(spins_mps)
for spin in spins_mps
    #println(v2energy(J, spin))
    binary = [Int(i > 0) for i in spin]
    println(binary)
    println(binary'*M*binary)
end


if file == "data/QUBO8qbits"
    println("testing ground")
    binary = [Int(i > 0) for i in spins_mps[1]]
    println(binary == [0,1,0,0,1,0,0,0])
elseif file == "data/QUBO6qbits"
    println("testing ground")
    binary = [Int(i > 0) for i in spins_mps[1]]
    println(binary == [0,1,0,1,0,0])
end
