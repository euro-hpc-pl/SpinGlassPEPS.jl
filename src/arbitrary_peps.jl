using LinearAlgebra
using TensorOperations


######## arbitrary decomposition  ############


function set_spins_on_mps(mps::Vector{Array{T, 5}}, s::Vector{Int}) where T <: AbstractFloat
    l = length(mps)
    up_bonds = zeros(Int, l)
    output_mps = Array{Union{Nothing, Array{T, 4}}}(nothing, l)
    for i in 1:l
        if s[i] == 0
            output_mps[i] = sum_over_last(mps[i])
        else
            A = set_last(mps[i], s[i])
            ind = spins2index(s[i])

            output_mps[i] = A

        end
    end
    Vector{Array{T, 4}}(output_mps)
end

function comp_marg_p(mps_u::Vector{Array{T, 4}}, mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)

    mps_n = MPSxMPO(mpo, mps_u)
    compute_scalar_prod(mps_d, mps_n), mps_n
end

function comp_marg_p_first(mps_d::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mps_u = set_spins_on_mps(M, ses)
    compute_scalar_prod(mps_d, mps_u), mps_u
end

function comp_marg_p_last(mps_u::Vector{Array{T, 4}}, M::Vector{Array{T, 5}}, ses::Vector{Int}) where T <: AbstractFloat
    mpo = set_spins_on_mps(M, ses)
    mps_u = copy(mps_u)

    compute_scalar_prod(mpo, mps_u)
end

mutable struct Partial_sol1{T <: AbstractFloat}
    spins::Vector{Int}
    objective::T
    upper_mps::Vector{Array{T, 4}}
    function(::Type{Partial_sol1{T}})(spins::Vector{Int}, objective::T, upper_mps::Vector{Array{T, 4}}) where T <:AbstractFloat
        new{T}(spins, objective, upper_mps)
    end
    function(::Type{Partial_sol1{T}})(spins::Vector{Int}, objective::T) where T <:AbstractFloat
        new{T}(spins, objective, [zeros(0,0,0,0)])
    end
    function(::Type{Partial_sol1{T}})() where T <:AbstractFloat
        new{T}(Int[], 1., [zeros(0,0,0,0)])
    end
end

function add_spin(ps::Partial_sol1{T}, s::Int) where T <: AbstractFloat
    s in [-1,1] || error("spin should be 1 or -1 we got $s")
    Partial_sol1{T}(vcat(ps.spins, [s]), T(0.), ps.upper_mps)
end


function solve_arbitrary_decomposition(qubo::Vector{Qubo_el{T}}, grid::Matrix{Int}, no_sols::Int = 2; β::T) where T <: AbstractFloat
    problem_size = maximum(grid)
    s = size(grid)
    M = make_pepsTN(grid, qubo, β)

    Partial_solutions = Partial_sol1{T}[Partial_sol1{T}()]

    for row in 1:s[1]
        #this may need to ge cashed
        # for the itterative optimisation normalisation
        # would be necessary
        lower_mps = make_lower_mps(M, row + 1, 0, 0.)

        for j in grid[row,:]

            a = [add_spin(ps, 1) for ps in Partial_solutions]
            b = [add_spin(ps, -1) for ps in Partial_solutions]
            Partial_solutions = vcat(a,b)

            for ps in Partial_solutions

                part_sol = ps.spins
                sol = part_sol[1+(row-1)*s[2]:end]
                l = s[2] - length(sol)
                sol = vcat(sol, fill(0, l))

                u = []
                prob = 0.
                if row == 1
                    prob, u = comp_marg_p_first(lower_mps, M[row,:], sol)
                elseif row == s[1]
                    prob = comp_marg_p_last(ps.upper_mps, M[row,:], sol)
                else
                    prob, u = comp_marg_p(ps.upper_mps, lower_mps, M[row,:], sol)
                end

                ps.objective = prob

                if (j % s[2] == 0) & (j < problem_size)
                    ps.upper_mps = u
                end
            end

            objectives = [ps.objective for ps in Partial_solutions]

            perm = sortperm(objectives)

            p1 = last_m_els(perm, no_sols)

            Partial_solutions = Partial_solutions[p1]

            if j == problem_size
                    return Partial_solutions
            end
        end
    end
end
