
"""
    function brute_force_solve(M::Matrix{T}, sols::Int) where T <: AbstractFloat

returns vector of solutions (Vector{Vector{Int}}) and vector of energies (Vector{T})

First element is supposed to be a ground state
"""
function brute_force_solve(M::Matrix{T}, sols::Int) where T <: AbstractFloat
    s = size(M,1)
    all_spins = Vector{Int}[]
    energies = Float64[]
    for i in 1:2^s
        spins = ind2spin(i, s)
        push!(all_spins, spins)
        energy = -v2energy(M, spins)
        push!(energies, energy)
    end
    p = sortperm(energies)
    all_spins[p][1:sols], energies[p][1:sols]
end

"""
    function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat

returne energy Float given a matrix of interacrions and vector of spins
"""
function v2energy(M::Matrix{T}, v::Vector{Int}) where T <: AbstractFloat
    d =  diag(M)
    M = M .- diagm(d)
    transpose(v)*M*v + transpose(v)*d
end
