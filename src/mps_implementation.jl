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

function Btensor(T::Type, d::Int)
    B_tensor = zeros(T, d,d,d,d)
    for i in 1:d
        @inbounds B_tensor[i,i,i,i] = T(1.)
    end
    # always most left
    return sum(B_tensor, dims = 1)
end

function Ctensor(T::Type, J::Float64, d::Int, most_right::Bool = false)

    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for k in 1:d
            # TODO assumed d = 2 otherwise correct
            ret[i,k,i,k] = exp(J*(2*i-3)*(2*k-3))
        end
    end
    if most_right
        return sum(ret, dims = 3)
    end
    ret
end

VE = Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int}}
function add_MPO!(mpo::MPO{T}, iij::VE, g::MetaGraph, β::T) where T<: AbstractFloat

    i = src(iij[1])
    d = length(props(g, i)[:energy])

    l = dst(iij[end])
    for j in src(iij[1]):dst(iij[end])
        mpo[j] = ones13_24(T, d)
    end

    mpo[i] = Btensor(T, d)
    for e in iij
        j = dst(e)
        # minus for convention
        J = -props(g, e)[:J]
        # minus for probability
        mpo[j] = Ctensor(T, -J*β, d, j==l)
    end
end


function add_phase!(mps::MPS{T}, g::MetaGraph, β::T) where T<: AbstractFloat

    for i in 1:length(mps)
        internal_e = props(g, i)[:energy]
        for j in 1:length(internal_e)
            mps[i][:,j,:] = mps[i][:,j,:]*exp(-β/2*internal_e[j])
        end
    end
end


function construct_mps_step(mps::MPS{T}, g::MetaGraph, β::T, edges_sets::VE) where T<: AbstractFloat

    phys_dims = size(mps[1], 2)

    mpo = MPO([ones24(T, phys_dims) for _ in 1:length(mps)])

    sources = [src(e) for e in edges_sets]
    for v in unique(sources)
         edges = filter(x -> src(x) == v, edges_sets)
         add_MPO!(mpo, edges, g, β)
    end
    mpo*mps
end


function connections_for_mps(g::MetaGraph)

     g1 = copy(g)
     connections = Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int}}[]
     for v in vertices(g1)
         es = edges(g1)
         pe = filter(x -> src(x) == v, collect(es))
         if pe != Int[]
             for j in v:nv(g1)
                 # to make mps elements to be disconected
                 if j > maximum([dst(e) for e in pe])
                     pe1 = filter(x -> src(x) == j, collect(es))
                     pe = vcat(pe, pe1)
                 end
             end
             push!(connections, pe)
         end
        for e in pe
            rem_edge!(g1, e)
        end
    end
    connections
end


function construct_mps(M::Matrix{Float64}, β::T, β_step::Int, χ::Int, threshold::Float64) where T<: AbstractFloat
    g = M2graph(M)
    g = graph4mps(g)

    construct_mps(g, β, β_step, χ, threshold)
end



function construct_mps(g::MetaGraph, β::T, β_step::Int, χ::Int, threshold::T) where T<: AbstractFloat

    v = connections_for_mps(g)

    l = nv(g)
    #TODO, get d from g
    d = 2
    mps = MPS([ones(T, 1,d,1) for _ in 1:l])
    for _ in 1:β_step
        for el in v
            mps = construct_mps_step(mps, g,  β/β_step, el)
            s = maximum([size(e, 1) for e in mps])
            if ((threshold > 0) * (s > χ))
                mps = compress(mps, χ, threshold)
            end
        end
    end
    add_phase!(mps, g, β)
    mps
end


"""
    contract4probability(A::Array{T,3}, M::Matrix{T}, v::Vector{T})

return vector diag(prob) the result of the following contraction

           v -- A .
                |   .
                      .
                      M
                      .
                |   .
           v -- A .

"""

function contract4probability(A::Array{T,3}, M::Matrix{T}, v::Vector{T}) where T <: AbstractFloat
    probs = zeros(T, size(A,2))
    for i in 1:size(A,2)
        @inbounds A1 = A[:,i,:]
        @inbounds probs[i] = transpose(v)*A1*M*transpose(A1)*v
    end
    return probs
end


function compute_probs(mps::MPS{T}, spins::Vector{Int}) where T <: AbstractFloat

    k = length(spins)
    mm = [mps[i][:,spins[i],:] for i in 1:k]
    left_v = Mprod(mm)[1,:]

    right_m = ones(T,1,1)
    if k+1 < length(mps)

        right_m = compute_scalar_prod(MPS(mps[k+2:end]), MPS(mps[k+2:end]))
    end
    contract4probability(mps[k+1], right_m, left_v)
end


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
