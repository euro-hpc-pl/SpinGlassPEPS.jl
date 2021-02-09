export return_solution


"""
    mutable struct Partial_sol{T <: Real}

structure of the partial solution
"""
mutable struct Partial_sol{T <: Real}
    spins::Vector{Int}
    objective::T

    function(::Type{Partial_sol{T}})(spins::Vector{Int}, objective::T) where T <:Real
        new{T}(spins, objective)
    end
    function(::Type{Partial_sol{T}})() where T <:Real
        new{T}(Int[], 1.)
    end
end


"""
    update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: Real

Add a spin and replace an objective function to Partial_sol{T} type
"""
# TODO move particular type to solver
function update_partial_solution(ps::Partial_sol{T}, s::Int, objective::T) where T <: Real
    Partial_sol{T}(vcat(ps.spins, [s]), objective)
end


function project_spin_from_above(projector::Array{T,2}, spin::Int64, mps_el::Array{T,3}) where T <: Real
    @reduce B[a, b] := sum(x) projector[$spin, x] * mps_el[a, x, b]
    B
end

function project_spin_from_above(projector::Array{T,2}, spin::Int64, mpo_el::Array{T,4}) where T <: Real
    @reduce B[a, d, c] := sum(x) projector[$spin, x] * mpo_el[a, x, c, d]
    B
end


"""
....


b_m - boundary mps
p_r - peps row

              physical .
                         .   upper_right
                           .  |      |
    upper_left   from_left-- p_r3 --  p_r4 -- 1
      |    |                 |      |
1 --b_m1 --b_m2       --     b_m  -- b_m -- 1
"""
function conditional_probabs(peps::PepsNetwork, ps::Partial_sol{T}, boundary_mps::MPS{T}, peps_row::PEPSRow{T}) where T <: Number


    j = length(ps.spins) + 1
    ng = peps.network_graph
    fg = ng.factor_graph


    k = j % peps.j_max
    if k == 0
        k = peps.j_max
    end
    row = ceil(Int, j/peps.j_max)

    mpo = MPO(peps_row)

    # set from above
    # not in last row
    if j > peps.j_max*(peps.i_max-1)
        spin = [1 for _ in 1:k-1]
    else
        spin = [ps.spins[i] for i in j-k+1:j-1]
    end

    proj_u = [projectors(fg, i, i+peps.j_max)[1] for i in j-k+1:j-1]


    BB = [project_spin_from_above(proj_u[i], spin[i], boundary_mps[i]) for i in 1:k-1]

    weight = ones(T, 1,1)
    if k > 1
        weight = prod(BB)
    end

    # set form left
    if k == 1
        spin = 1
    else
        spin = ps.spins[end]
    end

    proj_l, _, _ = projectors(fg, j-1, j)
    @reduce A[d, a, b, c] := sum(x) proj_l[$spin, x] * peps_row[$k][x, a, b, c, d]

    r = j-k+peps.j_max
    if j <= peps.j_max
        spin = [1 for _ in j:r]
    else
        spin = [ps.spins[i-peps.j_max] for i in j:r]
    end

    proj_u = [projectors(fg, i-peps.j_max, i)[1] for i in j:r]


    mpo = mpo[k+1:end]
    mpo = MPO(vcat([A], mpo))


    CC = [project_spin_from_above(proj_u[i], spin[i], mpo[i]) for i in 1:length(mpo)]

    upper_mps = MPS(CC)

    lower_mps = MPS(boundary_mps[k:end])

    re = right_env(lower_mps, upper_mps)[1]

    probs_unnormed = re*transpose(weight)

    objective = probs_unnormed./sum(probs_unnormed)

    dropdims(objective; dims = 2)
end


"""
    function dX_inds(grid::Matrix{Int}, j::Int; has_diagonals::Bool = false)

Returns vector{Int} indexing of the boundary region (dX) given a grid.
id has diagonals, diagonal bounds on the grid are taken into account
"""

function dX_inds(s::Int, j::Int; has_diagonals::Bool = false)
    last = j-1
    first = maximum([1, j - s])
    if (has_diagonals & (j%s != 1))
        first = maximum([1, j - s - 1])
    end
    return collect(first: last)
end

"""
    function merge_dX(partial_s::Vector{Partial_sol{T}}, dX_inds::Vector{Int}, δH::Float64) where T <:Real

Return a vector of Partial_sol{T}, with merged boundaries.

Merging rule is such that the retion of the objective function of the merged item
to the maximal is lower than δH
"""

function merge_dX(partial_s::Vector{Partial_sol{T}}, dX_inds::Vector{Int}, δH::Float64) where T <:Real
    if (length(partial_s) > 1) & (δH != .0)
        leave = [true for _ ∈ partial_s]

        dXes = [ps.spins[dX_inds] for ps ∈ partial_s]

        unique_dXes = unique(dXes)
        if dXes != unique_dXes
            dXcount = countmap(dXes)
            for dX ∈ unique_dXes
                if dXcount[dX] > 1
                    i = findall(k -> k == dX, dXes)
                    objectives = [partial_s[j].objective for j ∈ i]

                    objectives = objectives./maximum(objectives)
                    for ind ∈ i[objectives .< δH]
                        leave[ind] = false
                    end
                end
            end
            no_reduced = count(.!(leave))
            # this is just for testing
            if no_reduced > 0
                j = length(partial_s[1].spins)
                k = length(partial_s)
                println(no_reduced, " out of $k partial solutions deleted at j = $j")
            end
            return partial_s[leave]
        end
    end
    partial_s
end


function solve(peps::PepsNetwork, no_sols::Int = 2; β::T, χ::Int = 2^prod(node_size),
                                               threshold::Float64 = 0.,
                                               δH::Float64 = 0., max_sweeps=4) where T <: Real



    boundary_mps = boundaryMPS(peps, 2, χ, threshold, max_sweeps)

    partial_s = Partial_sol{T}[Partial_sol{T}()]
    for row ∈ 1:peps.i_max
        @info "row of peps = " row

        peps_row = PEPSRow(peps, row)

        a = (row-1)*peps.j_max

        for k ∈ 1:peps.j_max
            j = a + k

            dX = dX_inds(peps.j_max, j)

            partial_s_temp = Partial_sol{T}[]
            # TODO better compare energies, think it over
            partial_s = merge_dX(partial_s, dX, δH)
            for ps ∈ partial_s

                objectives = conditional_probabs(peps, ps, boundary_mps[row], peps_row)

                for l ∈ eachindex(objectives)
                    new_objectives = ps.objective*objectives[l]
                    # TODO use log of probabilities
                    ps1 = update_partial_solution(ps, l, new_objectives)
                    push!(partial_s_temp, ps1)
                end

            end
            partial_s = select_best_solutions(partial_s_temp, no_sols)

            if j == peps.i_max*peps.j_max

                return partial_s = partial_s[end:-1:1]

            end
        end
    end
end

function return_solution(g::MetaGraph{Int,T}, fg::MetaDiGraph{Int,T},
                                              partial_s::Vector{Partial_sol{T}}) where T <: Real

    sols = [Int[]]
    L = props(g)[:L]
    for ps in partial_s


        sol = zeros(Int, L)
        for k in 1:nv(fg)
            D = props(fg, k)[:cluster].vertices
            p = sortperm([e for e in values(D)])
            inds = [e for e in keys(D)][p]

            all_states = props(fg, k)[:spectrum].states
            sp = ps.spins[k]
            sol[inds] = all_states[sp]
        end
        push!(sols, sol)
    end

    return sols[2:end]
end


"""
    select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:Real

returns Vector{Partial_sol{T}}, a vector of no_sols best solutions
"""
function select_best_solutions(partial_s_temp::Vector{Partial_sol{T}}, no_sols::Int) where T <:Real
    obj = [ps.objective for ps ∈ partial_s_temp]
    # TODO change sortperm to partial sort
    perm = sortperm(obj)
    p = last_m_els(perm, no_sols)

    return partial_s_temp[p]
end
