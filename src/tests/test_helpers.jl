function form_peps(gg::MetaGraph, β::Float64, s::Tuple{Int, Int} = (3,3))
    M = Array{Union{Nothing, Array{Float64}}}(nothing, s)
    k = 0
    for i in 1:s[1]
        for j in 1:s[2]
            k = k+1
            M[i,j] = compute_single_tensor(gg, k, β)
        end
    end
    Matrix{Array{Float64, 5}}(M)
end


# NCon constraction
function contract3x3by_ncon(M::Matrix{Array{Float64, 5}})
    u1 = M[1,1][1,1,:,:,:]
    v1 = [2,31, -1]


    u2 = M[1,2][:,1,:,:,:]
    v2 = [2,3,32,-2]

    u3 = M[1,3][:,1,1,:,:]
    v3 = [3,33,-3]

    m1 = M[2,1][1,:,:,:,:]
    v4 = [31,  4, 41, -4]

    m2 = M[2,2]
    v5 = [4, 32, 5, 42, -5]

    m3 = M[2,3][:,:,1,:,:]
    v6 = [5, 33, 43, -6]

    d1 = M[3,1][1,:,:,1,:]
    v7 = [41, 6, -7]

    d2 = M[3,2][:,:,:,1,:]
    v8 = [6,42,7,-8]

    d3 = M[3,3][:,:,1,1,:]
    v9 = [7, 43, -9]

    tensors = (u1, u2, u3, m1, m2, m3, d1, d2, d3)
    indexes = (v1, v2, v3, v4, v5, v6, v7, v8, v9)

    ncon(tensors, indexes)
end

"""
    function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat

return energy Float given a matrix of interacrions and vector of spins
"""
function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)
    -transpose(v)*M*v - transpose(v)*d
end

function make_interactions_case1()
    L = 9

    D = Dict{Tuple{Int64,Int64},Float64}()
    push!(D, (1, 1) => .5)
    push!(D, (1, 2) => -0.5)
    push!(D, (1, 4) => -1.5)
    push!(D, (2, 2) => -1.)
    push!(D, (2, 3) => -1.5)
    push!(D, (2,5) => -0.5)
    push!(D, (3,3) => 2.)
    push!(D, (3,6) => 1.5)
    push!(D, (5,6) => -0.25)
    push!(D, (4,5) => 0.5)
    push!(D, (6,6) =>  .05)
    push!(D, (6,9) => -0.52)
    push!(D, (5,5) => 0.75)
    push!(D, (5,8) => 0.5)
    push!(D, (4,4) => 0.)
    push!(D, (4,7) => -0.01)
    push!(D, (7,7) => 0.35)
    push!(D, (7,8) => 0.7)
    push!(D, (8,8) => -0.08)
    push!(D, (8,9) => -0.05)
    push!(D, (9,9) => 0.33)

    ising_graph(D, L, 1, -1)
end


function interactions_case2()
    css = -2.
    D = Dict{Tuple{Int64,Int64},Float64}()
    push!(D, (1,1) => -1.25)
    push!(D, (1,2) => 1.75)
    push!(D, (1,4) => css)
    push!(D, (2,2) => -1.75)
    push!(D, (2,3) => 1.75)
    push!(D, (2,5) => 0.)
    push!(D, (3,3) => -1.75)
    push!(D, (3,6) => css)
    push!(D, (5,6) => 1.75)
    push!(D, (4,5) => 1.75)
    push!(D, (6,6) => 0.)
    push!(D, (6,9) => 0.)
    push!(D, (5,5) => -0.75)
    push!(D, (5,8) => 0.)
    push!(D, (4,4) => 0.)
    push!(D, (4,7) => 0.)
    push!(D, (7,7) => css)
    push!(D, (7,8) => 0.)
    push!(D, (8,8) => css)
    push!(D, (8,9) => 0.)
    push!(D, (9,9) => css)

    L = 9

    ising_graph(D, L, 1, -1)
end
