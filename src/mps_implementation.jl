include("peps_no_types.jl")
include("notation.jl")


function scalar_prod_with_itself(mps::Vector{Array{T, 3}}) where T <: AbstractFloat
    env = ones(T, 1,1)
    for i in length(mps):-1:1
        env = scalar_prod_step(mps[i], env)
    end
    env
end

function scalar_prod_step(mps::Array{T, 3}, env::Array{T, 2}) where T <: AbstractFloat
    C = zeros(T, size(mps, 1), size(mps, 1))

    @tensor begin
        C[a,b] = mps[a,x,z]*mps[b,y,z]*env[x,y]
    end
    C
end


function initialize_mps(l::Int, physical_dims::Int =  2, T::Type = Float64)
    [ones(T, 1,1,physical_dims) for _ in 1:l]
end

function initialize_mpo(l::Int, physical_dims::Int =  2, T::Type = Float64)
    [make_ones_between_not_interactiong(physical_dims, T) for _ in 1:l]
end


####


# TODO this two below perhaps can be simplified


function make_ones_between_not_interactiong(d::Int, T::Type = Float64)
    ret = zeros(T, d,d)
    for j in 1:d
        ret[j,j] = T(1.)
    end
    reshape(ret, (1,1,d,d))
end

function make_ones_between_interacting_nodes(d::Int, T::Type = Float64)
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            ret[i,i,j,j] = T(1.)
        end
    end
    ret
end

### below the constriction of B-tensors and C-tensors
### TODO these should be tested directly in unit tests

function Btensor(d::Int, most_left::Bool = false, most_right::Bool = false, T::Type = Float64)
    B_tensor = zeros(T, d,d,d,d)
    for i in 1:d
        @inbounds B_tensor[i,i,i,i] = T(1.)
    end
    if most_left
        return sum(B_tensor, dims = 1)
    elseif most_right
        return sum(B_tensor, dims = 2)
    end
    return B_tensor
end

function Ctensor(J::Vector{T}, d::Int, most_left::Bool = false, most_right::Bool = false) where T <: AbstractFloat

    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for k in 1:d
            # TODO this need to be checked
            Jb = sum(J.*ind2spin(i).*ind2spin(k))
            ret[i,i,k,k] = exp(Jb)
        end
    end
    if most_left
        return sum(ret, dims = 1)
    elseif most_right
        return sum(ret, dims = 2)
    end
    return ret
end

function add_MPO!(mpo::Vector{Array{T, 4}}, i::Int, nodes::Vector{Int},
                    g::MetaGraph, β::T) where T<: AbstractFloat

    d = 2
    if props(g, 1)[:internal_struct] != Dict()
        # TODO, correct it
        d = 2
    end

    k = minimum([i, nodes...])
    l = maximum([i, nodes...])
    for j in k:l
        mpo[j] = make_ones_between_interacting_nodes(d, T)
    end
    mpo[i] = Btensor(d, i==k, i==l, T)
    for j in nodes
        # TODO remove array
        J = [props(g, Edge(i,j))[:J]]

        mpo[j] = Ctensor(J*β, d, j==k, j==l)
    end
    mpo
end

# TODO interactions are not necessary
function add_phase!(mps::Vector{Array{T, 3}}, g::MetaGraph, β::T) where T<: AbstractFloat

    d = size(mps[1], 3)
    for i in 1:length(mps)
        internal_e = props(g, i)[:log_energy]
        for j in 1:d
            mps[i][:,:,j] = mps[i][:,:,j]*exp(internal_e[j]*β/2)
        end
    end
end


function set_part_of_spins(mps::Vector{Array{T, 3}}, spins::Vector{Int}) where T <: AbstractFloat
    env = ones(T,1)
    for i in 1:length(spins)
        env = env*mps[i][:,:,spins[i]]
    end
    reshape(env, size(env,2))
end

function compute_probs(mps::Vector{Array{T, 3}}, spins::Vector{Int}) where T <: AbstractFloat
    d = size(mps[1], 3)
    k = length(spins)+1
    left_v = set_part_of_spins(mps, spins)

    A = mps[k]
    probs_at_k = zeros(T, d,d)
    if k < length(mps)
        right_m = scalar_prod_with_itself(mps[k+1:end])

        @tensor begin
            probs_at_k[x,y] = A[a,b,x]*A[c,d,y]*left_v[a]*left_v[c]*right_m[b,d]
        end

    else
        # uses one() insted of right mstrix
        @tensor begin
            probs_at_k[x,y] = A[a,b,x]*A[c,b,y]*left_v[a]*left_v[c]
        end
    end
    return(diag(probs_at_k))
end

function construct_mps_step(mps::Vector{Array{T, 3}}, g::MetaGraph,
                                                    β::T, is::Vector{Int},
                                                    js::Vector{Vector{Int}}) where T<: AbstractFloat
    ##TODO physical dimention may be differet
    phys_dims = [size(el, 3) for el in mps]
    mpo = [make_ones_between_not_interactiong(phys_dims[i]) for i in 1:length(mps)]
    for k in 1:length(is)
        add_MPO!(mpo, is[k], js[k], g, β)
    end
    MPSxMPO(mps, mpo)
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

function split_if_differnt_spins(is::Vector{Int}, js::Vector{Vector{Int}}, ns::Vector{Node_of_grid})
    is_new = Int[]
    js_new = Vector{Int}[]
    for i in 1:length(is)
        el_i = is[i]
        el_j = js[i]
        n = ns[el_i].connected_nodes
        a = findall(x -> x == el_j[1], n)[1]

        spins = ns[el_i].connected_spins[a][:,1]
        push!(is_new, el_i)
        j_temp = Int[]
        for j in el_j
            a = findall(x -> x == j, n)[1]
            spins_temp = ns[el_i].connected_spins[a][:,1]

            if spins != spins_temp
                push!(is_new, el_i)
                push!(js_new, j_temp)
                j_temp = Int[]
                spins = spins_temp
            end
            push!(j_temp, j)
        end
        push!(js_new, j_temp)
    end
    is_new, js_new
end


function construct_mps(M::Matrix{T}, β::T, β_step::Int, χ::Int, threshold::T) where T<: AbstractFloat
    ints = M2interactions(M)
    g = interactions2graph(ints)
    
    construct_mps(g, β, β_step, χ, threshold)
end

function construct_mps(g::MetaGraph, β::T, β_step::Int, χ::Int, threshold::T) where T<: AbstractFloat

    is, js = connections_for_mps(g)
    #is,js = split_if_differnt_spins(is,js,ns)
    all_is, all_js = cluster_conncetions(is,js)

    l = nv(g)
    mps = initialize_mps(l)
    for _ in 1:β_step
        for k in 1:length(all_is)
            mps = construct_mps_step(mps, g,  β/β_step, all_is[k], all_js[k])

            s = maximum([size(e, 1) for e in mps])
            if ((threshold > 0) * (s > χ))
                mps = compress_iter(mps, χ, threshold)
                #mps = compress_svd(mps, χ)
            end
        end
    end
    add_phase!(mps, g, β)
    mps
end

### TODO should be reintegrated with the solve
function solve_mps(g::MetaGraph, no_sols::Int; β::T, β_step::Int, χ::Int = 0, threshold::T = T(0.)) where T <: AbstractFloat

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
