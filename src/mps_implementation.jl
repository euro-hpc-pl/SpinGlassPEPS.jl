include("peps.jl")
include("notation.jl")

function initialize_mps(l::Int, physical_dims::Int =  2, T::Type = Float64)
    [ones(T, 1,1,physical_dims) for _ in 1:l]
end

function initialize_mpo(l::Int, physical_dims::Int =  2, T::Type = Float64)
    [make_ones(T) for _ in 1:l]
end


function make_ones(T::Type = Float64)
    d = 2
    ret = zeros(T, 1,1,d,d)
    for j in 1:d
        ret[1,1,j,j] = T(1.)
    end
    ret
end

function make_ones_inside(T::Type = Float64)
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            ret[i,i,j,j] = T(1.)
        end
    end
    ret
end

function T_with_B(l::Bool = false, r::Bool = false, T::Type = Float64)
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for l in 1:d
                    ret[i,j,k,l] = T(i==j)*T(k==l)*T(j==l)
                end
            end
        end
    end
    if l
        return sum(ret, dims = 1)
    elseif r
        return sum(ret, dims = 2)
    end
    return ret
end

function T_with_C(Jb::T, l::Bool = false, r::Bool = false) where T <: AbstractFloat
    d = 2
    ret = zeros(T, d,d,d,d)
    for i in 1:d
        for j in 1:d
            for k in 1:d
                for l in 1:d
                    ret[i,j,k,l] = T(i==j)*T(k==l)*exp(Jb*ind2spin(i)*ind2spin(k))
                end
            end
        end
    end
    if l
        return sum(ret, dims = 1)
    elseif r
        return sum(ret, dims = 2)
    end
    return ret
end

function add_MPO!(mpo::Vector{Array{T, 4}}, i::Int, i_n::Vector{Int}, qubo::Vector{Qubo_el{T}}, β::T) where T<: AbstractFloat
    k = minimum([i, i_n...])
    l = maximum([i, i_n...])
    for j in k:l
        mpo[j] = make_ones_inside(T)
    end
    mpo[i] = T_with_B(i==k, i==l, T)
    for j in i_n
        J = JfromQubo_el(qubo, i,j)
        mpo[j] = T_with_C(J*β, j==k, j==l)
    end
    mpo
end

function add_phase!(mps::Vector{Array{T, 3}}, qubo::Vector{Qubo_el{T}}, β::T) where T<: AbstractFloat
    d = size(mps[1], 3)
    for i in 1:length(mps)
        # we have a twice from the scalar product
        h = JfromQubo_el(qubo, i,i)/2
        for j in 1:d
            mps[i][:,:,j] = mps[i][:,:,j]*exp(ind2spin(j)*β*h)
        end
    end
end

function v_from_mps(mps::Vector{Array{T, 3}}, spins::Vector{Int}) where T <: AbstractFloat
    env = ones(T,1)
    for i in 1:length(spins)
        j = spins2ind(spins[i])
        env = env*mps[i][:,:,j]
    end
    reshape(env, size(env,2))
end

function compute_probs(mps::Vector{Array{T, 3}}, spins::Vector{Int}) where T <: AbstractFloat
    d = size(mps[1], 3)
    k = length(spins)+1
    left_v = v_from_mps(mps, spins)

    A = mps[k]
    probs_at_k = zeros(T, d,d)
    if k < length(mps)
        right_m = compute_scalar_prod(mps[k+1:end], mps[k+1:end])

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

function construct_mps_step(mps::Vector{Array{T, 3}}, qubo::Vector{Qubo_el{T}},
                                                    β::T, is::Vector{Int},
                                                    js::Vector{Vector{Int}}) where T<: AbstractFloat
    mpo = [make_ones() for _ in 1:length(mps)]
    for k in 1:length(is)
        add_MPO!(mpo, is[k], js[k] ,qubo, β)
    end
    MPSxMPO(mps, mpo)
end


function construct_mpo_step(mpo_d::Vector{Array{T, 4}}, qubo::Vector{Qubo_el{T}},
                                                    β::T, is::Vector{Int},
                                                    js::Vector{Vector{Int}}) where T<: AbstractFloat
    mpo = [make_ones() for _ in 1:length(mpo_d)]
    for k in 1:length(is)
        add_MPO!(mpo, is[k], js[k] ,qubo, β)
    end
    MPOxMPO(mpo_d, mpo)
end

function construct_mps1(qubo::Vector{Qubo_el{T}}, β::T, β_step::Int, l::Int,
                                                all_is::Vector{Vector{Int}},
                                                all_js::Vector{Vector{Vector{Int}}},
                                                χ::Int, threshold::T) where T<: AbstractFloat

    mps = initialize_mps(l)
    for k in 1:length(all_is)
        mps = construct_mps_step(mps, qubo, β/β_step, all_is[k], all_js[k])
        s = maximum([size(e, 1) for e in mps])
        if ((threshold > 0) * ((s > 2*χ) | k==length(all_is)))
            mps = compress_iter(mps, χ, threshold)
        end
    end

    for a in 2:β_step
        mpo = initialize_mpo(l)
        for k in 1:length(all_is)
            mpo = construct_mpo_step(mpo, qubo, β/β_step, all_is[k], all_js[k])
            s = maximum([size(e, 1) for e in mpo])
            if ((threshold > 0) * ((s > 2*χ) | k==length(all_is)))
                mpo = compress_iter(mpo, χ, threshold)
            end
        end
        mps = MPSxMPO(mps, mpo)
        if threshold > 0
            mps = compress_iter(mps, χ, threshold)
        end
        println(a)
    end
    add_phase!(mps,qubo, β)
    println([size(e) for e in mps])
    mps
end


function construct_mps(qubo::Vector{Qubo_el{T}}, β::T, β_step::Int, l::Int,
                                                all_is::Vector{Vector{Int}},
                                                all_js::Vector{Vector{Vector{Int}}},
                                                χ::Int, threshold::T) where T<: AbstractFloat
    mps = initialize_mps(l)
    for a in 1:β_step
        for k in 1:length(all_is)
            mps = construct_mps_step(mps, qubo, β/β_step, all_is[k], all_js[k])

            s = maximum([size(e, 1) for e in mps])
            if ((threshold > 0) * (s > 2*χ))
                mps = compress_iter(mps, χ, threshold)
            end
        end
        println(a)
    end
    add_phase!(mps,qubo, β)
    println([size(e) for e in mps])
    mps
end

function solve_mps(qubo::Vector{Qubo_el{T}}, all_is::Vector{Vector{Int}},
                all_js::Vector{Vector{Vector{Int}}}, problem_size::Int,
                no_sols::Int; β::T, β_step::Int, χ::Int = 0, threshold::T = T(0.)) where T <: AbstractFloat
    mps = construct_mps1(qubo, β, β_step, problem_size, all_is, all_js, χ, threshold)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for i in 1:problem_size

        partial_s_temp = Partial_sol{T}[Partial_sol{T}()]
        for ps in partial_s

            objectives = compute_probs(mps, ps.spins)

            a = add_spin_marginal(ps, -1, objectives[1])
            b = add_spin_marginal(ps, 1, objectives[2])

            if partial_s_temp[1].spins == []
                partial_s_temp = vcat(a,b)
            else
                partial_s_temp = vcat(partial_s_temp, a,b)
            end
        end

        obj = [ps.objective for ps in partial_s_temp]
        perm = sortperm(obj)
        p = last_m_els(perm, no_sols)
        partial_s = partial_s_temp[p]

        if i == problem_size
            # sort from the ground state
            return partial_s[end:-1:1]
        end
    end
end
