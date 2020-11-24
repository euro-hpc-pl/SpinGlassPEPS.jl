"""
    contract2probability(A::Array{T,3}, M::Matrix{T}, v::Vector{T})

return vector diag(prob) the result of the following contraction

           v -- A .
                |   .
                      .
                      M
                      .
                |   .
           v -- A .

"""

function contract2probability(A::Array{T,3}, M::Matrix{T}, v::Vector{T}) where T <: AbstractFloat
    #probs = zeros(T, size(A,2), size(A,2))
    #@tensor begin
    #    probs[x,y] = A[a,x,b]*A[c,y,d]*v[a]*v[c]*M[b,d]
    #end
    probs = zeros(T, size(A,2))
    for i in 1:size(A,2)
        @inbounds A1 = A[:,i,:]
        @inbounds probs[i] = transpose(v)*A1*M*transpose(A1)*v
    end
    #println(diag(probs)-probs1)
    return probs
end


"""
    ones24(T::Type, phys_dim::Int)

returns Array{T,4} of size = (1,phys_dim,1,phys_dim)
a[1,i,1,j] = 1. only if i = j and 0. otherwise

pass interactions between mode 2 and 4
"""
function ones24(T::Type, phys_dim::Int)
    d = (1,phys_dim,1,phys_dim)
    M = diagm(ones(T, phys_dim))
    reshape(M,d)
end

"""
    ones13_24(T::Type, phys_dim::Int)

returns Array{T,4} of size = (phys_dim,phys_dim,phys_dim,phys_dim)
a[i,j,k,l] = 1. only if i = k and j = l
otherwise a[i,j,k,l] = 0.

pass interactions between mode 2 and 4 as well as beteen mode 1 and 3
"""
function ones13_24(T::Type, phys_dim::Int)
    M = diagm(ones(T, phys_dim))
    M = kron(M,M)
    d = (phys_dim,phys_dim,phys_dim,phys_dim)
    reshape(M,d)
end

### below the constriction of B-tensors and C-tensors

function Btensor(T::Type, d::Int, most_left::Bool = false, most_right::Bool = false)
    B_tensor = zeros(T, d,d,d,d)
    for i in 1:d
        @inbounds B_tensor[i,i,i,i] = T(1.)
    end
    if most_left
        return sum(B_tensor, dims = 1)
    elseif most_right
        return sum(B_tensor, dims = 3)
    end
    return B_tensor
end

function Ctensor(T::Type, J::Float64, d::Int, most_left::Bool = false, most_right::Bool = false)

    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for k in 1:d
            # TODO assumed d = 2 otherwise correct
            ret[i,k,i,k] = exp(J*(2*i-3)*(2*k-3))
        end
    end
    if most_left
        return sum(ret, dims = 1)
    elseif most_right
        return sum(ret, dims = 3)
    end
    return ret
end

function add_MPO!(mpo::MPO{T}, i::Int, nodes::Vector{Int}, g::MetaGraph, β::T) where T<: AbstractFloat

    d = length(props(g, i)[:energy])

    k = minimum([i, nodes...])
    l = maximum([i, nodes...])
    for j in k:l
        mpo[j] = ones13_24(T, d)
    end
    mpo[i] = Btensor(T, d, i==k, i==l)
    for j in nodes
        # minus for convention
        J = -props(g, Edge(i,j))[:J]
        # minus for probability
        mpo[j] = Ctensor(T, -J*β, d, j==k, j==l)
    end
end


function add_phase!(mps::MPS{T}, g::MetaGraph, β::T) where T<: AbstractFloat

    for i in 1:length(mps)
        internal_e = props(g, i)[:energy]
        # usually length = 2
        for j in 1:length(internal_e)
            mps[i][:,j,:] = mps[i][:,j,:]*exp(-β/2*internal_e[j])
        end
    end
end


function compute_probs(mps::MPS{T}, spins::Vector{Int}) where T <: AbstractFloat

    k = length(spins)
    mm = [mps[i][:,spins[i],:] for i in 1:k]
    left_v = Mprod(mm)[1,:]

    right_m = ones(T,1,1)
    if k+1 < length(mps)
        # TODO should be simpyfied
        mps1 = MPS([mps[i] for i in k+2:length(mps)])
        right_m = compute_scalar_prod(mps1, mps1)
    end
    contract2probability(mps[k+1], right_m, left_v)
end

function construct_mps_step(mps::MPS{T}, g::MetaGraph, β::T, is::Vector{Int},
                                                    js::Vector{Vector{Int}}) where T<: AbstractFloat

    phys_dims = size(mps[1], 2)

    mpo = MPO([ones24(T, phys_dims) for _ in 1:length(mps)])

    for k in 1:length(is)
        add_MPO!(mpo, is[k], js[k], g, β)
    end
    mpo*mps
end

function cluster_conncetions(all_is::Vector{Int}, all_js::Vector{Vector{Int}})
    l = length(all_is)
    mins = [0 for _ in 1:l]
    maxes = [0 for _ in 1:l]

    for i in 1:l
        temp = [all_is[i], all_js[i]...]
        mins[i] = minimum(temp)
        maxes[i] = maximum(temp)
        #println(i)
    end
    is = Vector{Int}[]
    js = Vector{Vector{Int}}[]

    i2be_chosen = copy(all_is)

    for k in 1:100
        is_temp = Int[]
        js_temp = Vector{Int}[]
        max = 0
        min = 100
        for i in 1:l
            temp = [all_is[i], all_js[i]...]
            a = maximum(temp)
            b = minimum(temp)

            min_cond = (b > max) | (a < min)

            if min_cond && i2be_chosen[i] != 0
                push!(is_temp, all_is[i])
                push!(js_temp, all_js[i])

                i2be_chosen[i] = 0
                max = maximum([a, max])
                min = minimum([b, min])
            end
        end
        push!(is, is_temp)
        push!(js, js_temp)

        if maximum(i2be_chosen) == 0

            return is, js
        end
    end
end

function connections_for_mps(g::MetaGraph)
    all_is = Int[]
    all_js = Vector{Int}[]
    conections = Vector{Int}[]
    # TODO this should be corrected and made more clear
    for i in vertices(g)

        pairs = [[i, node] for node in all_neighbors(g, i)]
        pairs_not_accounted = Int[]
        for p in pairs

            if !(p in conections) && !(p[2:-1:1] in conections)
                push!(pairs_not_accounted, p[2])
                push!(conections, p)
            end
        end
        if pairs_not_accounted != Int[]
            push!(all_is, i)
            push!(all_js, pairs_not_accounted)
        end
    end
    all_is, all_js
end

#function split_if_differnt_spins(is::Vector{Int}, js::Vector{Vector{Int}}, ns::Vector{Node_of_grid})
#    is_new = Int[]
#    js_new = Vector{Int}[]
#    for i in 1:length(is)
#        el_i = is[i]
#        el_j = js[i]
#        n = ns[el_i].connected_nodes
#        a = findall(x -> x == el_j[1], n)[1]

#        spins = ns[el_i].connected_spins[a][:,1]
#        push!(is_new, el_i)
#        j_temp = Int[]
#        for j in el_j
#            a = findall(x -> x == j, n)[1]
#            spins_temp = ns[el_i].connected_spins[a][:,1]

#            if spins != spins_temp
#                push!(is_new, el_i)
#                push!(js_new, j_temp)
#                j_temp = Int[]
#                spins = spins_temp
#            end
#            push!(j_temp, j)
#        end
#        push!(js_new, j_temp)
#    end
#    is_new, js_new
#end


function construct_mps(M::Matrix{Float64}, β::T, β_step::Int, χ::Int, threshold::Float64) where T<: AbstractFloat
    g = M2graph(M)
    g = graph4mps(g)

    construct_mps(g, β, β_step, χ, threshold)
end

function construct_mps(g::MetaGraph, β::T, β_step::Int, χ::Int, threshold::T) where T<: AbstractFloat

    is, js = connections_for_mps(g)
    #is,js = split_if_differnt_spins(is,js,ns)
    all_is, all_js = cluster_conncetions(is,js)

    l = nv(g)
    d = 2
    mps = MPS([ones(T, 1,d,1) for _ in 1:l])
    for _ in 1:β_step
        for k in 1:length(all_is)
            mps = construct_mps_step(mps, g,  β/β_step, all_is[k], all_js[k])
            s = maximum([size(e, 1) for e in mps])
            if ((threshold > 0) * (s > χ))
                mps = compress(mps, χ, threshold)
            end
        end
    end
    add_phase!(mps, g, β)
    mps
end

### TODO should be reintegrated with the solve
function solve_mps(g::MetaGraph, no_sols::Int; β::T, β_step::Int, χ::Int = 0, threshold::T = T(0.)) where T <: AbstractFloat

    g = graph4mps(g)
    problem_size = nv(g)

    mps = construct_mps(g, β, β_step, χ, threshold)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for j in 1:problem_size

        partial_s_temp = Partial_sol{T}[]
        for ps in partial_s
            objectives = compute_probs(mps, ps.spins)

            for l in 1:length(objectives)
                push!(partial_s_temp, update_partial_solution(ps, l, objectives[l]))
            end
        end

        partial_s = select_best_solutions(partial_s_temp, no_sols)

        if j == problem_size
            return return_solutions(partial_s, g)
        end
    end
end
