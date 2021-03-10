

function M2graph(M::Matrix{Float64}, sgn::Int = 1)
    size(M,1) == size(M,2) || error("matrix not squared")
    L = size(M,1)
    #TODO we do not require symmetric, is it ok?

    D = Dict{Tuple{Int64,Int64},Float64}()
    for j ∈ 1:size(M, 1)
        for i ∈ 1:j
            if (i == j)
                push!(D, (i,j) => M[j,i])
            elseif M[j,i] != 0.
                push!(D, (i,j) => M[i,j]+M[j,i])
            end
        end
    end
    ising_graph(D, L, sgn)
end

"""
    last_m_els(vector::Vector{Int}, m::Int)

returns last m element of the Vector{Int} or the whole vector if it has less than m elements

"""
function last_m_els(vector::Vector{Int}, m::Int)
    if length(vector) <= m
        return vector
    else
        return vector[end-m+1:end]
    end
end

spins2binary(spins::Vector{Int}) = [Int(i > 0) for i ∈ spins]

binary2spins(spins::Vector{Int}) = [2*i-1 for i ∈ spins]
