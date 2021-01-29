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


function make_interactions_case1()
    L = 9

    D = Dict{Tuple{Int64,Int64},Float64}()
    push!(D, (1, 1) => 1.0)
    push!(D, (1, 2) => -1.0)
    push!(D, (1, 4) => -3)
    push!(D, (2, 2) => -2.0)
    push!(D, (2, 3) => -3.0)
    push!(D, (2,5) => -2.0)
    push!(D, (3,3) => 4.0)
    push!(D, (3,6) => 3.0)
    push!(D, (5,6) => -0.5)
    push!(D, (4,5) => 1.0)
    push!(D, (6,6) =>  0.1)
    push!(D, (6,9) => -1.04)
    push!(D, (5,5) => 1.5)
    push!(D, (5,8) => 1.0)
    push!(D, (4,4) => 0.0)
    push!(D, (4,7) => -0.02)
    push!(D, (7,7) => 0.7)
    push!(D, (7,8) => 1.7)
    push!(D, (8,8) => -0.16)
    push!(D, (8,9) => -0.1)
    push!(D, (9,9) => 0.66)

    ising_graph(D, L)#, ising_graph(D, L)
end


function make_interactions_case2()
    f = 1
    f1 = 1
    L = 16
    D1 =  Dict((1, 1) => -2.8, (1, 2) => 0.3, (1, 5) => 0.2, (2, 2) => 2.7, (2, 3) => 0.255, (2, 6) => 0.21, (3, 3) => -2.6)
    D12 = Dict((3, 4) => 0.222, (3, 7) => 0.213, (4, 4) => 2.5, (4, 8) => 0.2)
    D2 =  Dict((5, 5) => -2.4, (5, 6) => 0.15, (5, 9) => 0.211, (6, 6) => 2.3, (6, 7) => 0.2)
    D22 = Dict((6, 10) => 0.15, (7, 7) => -2.2, (7, 8) => 0.11, (7, 11) => 0.35, (8, 8) => 2.1, (8, 12) => 0.19)
    D3 =  Dict((9, 9) => -2.0, (9, 10) => 0.222, (9, 13) => 0.15, (10, 10) => 1.9, (10, 11) => 0.28)
    D32 = Dict((10, 14) => 0.21, (11, 11) => -1.8, (11, 12) => 0.19, (11, 15) => 0.18, (12, 12) => 1.7, (12, 16) => 0.27)
    D4 =  Dict((13, 13) => -1.6, (13, 14) => 0.32, (14, 14) => 1.5, (14, 15) => 0.19, (15, 15) => -1.4, (15, 16) => 0.21, (16, 16) => 1.3)
    D = merge!(+, D1, D12, D2, D22, D3, D32, D4)

    ising_graph(D, L)
end
