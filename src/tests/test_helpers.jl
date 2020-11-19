function form_peps(gg::MetaGraph, β::Float64, s::Tuple{Int, Int} = (3,3))
    M = Array{Union{Nothing, Array{Float64}}}(nothing, s)
    k = 0
    for i in 1:s[1]
        for j in 1:s[2]
            k = k+1
            t = compute_single_tensor(gg, k, β)
            if i == s[1]
                 t = permutedims(t, (1,3,2,4))
             else
                 t = permutedims(t, (1,3,2,4,5))
             end
            M[i,j] = t
        end
    end
    Matrix{Array{Float64, N} where N}(M)
end


# NCon constraction
function contract3x3by_ncon(M::Matrix{Array{T, N} where N}) where T <: AbstractFloat
    u1 = M[1,1][1,:,1,:,:]
    v1 = [2,31, -1]


    u2 = M[1,2][:,:,1,:,:]
    v2 = [2,3,32,-2]

    u3 = M[1,3][:,1,1,:,:]
    v3 = [3,33,-3]

    m1 = M[2,1][1,:,:,:,:]
    v4 = [4,  31, 41, -4]

    m2 = M[2,2]
    v5 = [4, 5, 32, 42, -5]

    m3 = M[2,3][:,1,:,:,:]
    v6 = [5, 33, 43, -6]

    d1 = M[3,1][1,:,:,:]
    v7 = [6, 41, -7]

    d2 = M[3,2][:,:,:,:]
    v8 = [6,7,42,-8]

    d3 = M[3,3][:,1,:,:]
    v9 = [7, 43, -9]

    tensors = (u1, u2, u3, m1, m2, m3, d1, d2, d3)
    indexes = (v1, v2, v3, v4, v5, v6, v7, v8, v9)

    ncon(tensors, indexes)
end
