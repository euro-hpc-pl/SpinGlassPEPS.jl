export all_states,
    local_basis,
    gibbs_tensor,
    brute_force,
    full_spectrum,
    Spectrum,
    idx,
    local_basis,
    energy,
    matrix_to_integers

@inline idx(σ::Int) = (σ == -1) ? 1 : σ + 1
@inline local_basis(d::Int) = union(-1, 1:d-1)
all_states(rank::Union{Vector,NTuple}) = Iterators.product(local_basis.(rank)...)

const State = Vector{<:Integer}

"""
$(TYPEDSIGNATURES)

A `Spectrum` represents the energy spectrum of a system.

A `Spectrum` consists of energy levels, their corresponding states, and integer representations of the states.

# Fields:
- `energies::Vector{<:Real}`: An array of energy levels.
- `states::AbstractArray{State}`: An array of states.
- `states_int::Vector{Int}`: An array of integer representations of states.

# Constructors:
- `Spectrum(energies, states, states_int)`: Creates a `Spectrum` object with the specified energy levels, states, and integer representations.
- `Spectrum(energies, states)`: Creates a `Spectrum` object with the specified energy levels and states, automatically generating integer representations.
"""
struct Spectrum
    energies::Vector{<:Real}
    states::AbstractArray{State}
    states_int::Vector{Int}
    function Spectrum(energies, states, states_int)
        new(energies, states, states_int)
    end
    function Spectrum(energies, states::Matrix)
        states_int = matrix_to_integers(states)
        new(energies, Vector{eltype(states)}[eachcol(states)...], states_int)
    end
    function Spectrum(energies, states::Vector{<:State})
        states_int = matrix_to_integers(states)
        new(energies, states, states_int)
    end
end

"""
$(TYPEDSIGNATURES)

Converts a matrix of binary vectors to their integer representations.

This function takes a matrix of binary vectors, where each row represents a binary vector, and converts them into their corresponding integer representations.

# Arguments:
- `matrix::Vector{Vector{T}}`: A matrix of binary vectors.

# Returns:
- `Vector{Int}`: An array of integer representations of the binary vectors.
"""
function matrix_to_integers(matrix::Vector{<:Vector{<:Integer}})
    nrows = length(matrix[1])
    multipliers = 2 .^ collect(0:nrows-1)
    div.((hcat(matrix...)' .+ 1), 2) * multipliers
end

function matrix_to_integers(matrix::Matrix)
    nrows = size(matrix, 1)
    multipliers = 2 .^ collect(0:nrows-1)
    div.(matrix' .+ 1, 2) * multipliers
end

"""
    energy(σ::Vector, ig::IsingGraph)

Calculates the energy of a state in an Ising graph.

This function calculates the energy of a given state in the context of an Ising graph. 
The energy is computed based on the interactions between spins and their associated biases.

# Arguments:
- `σ::AbstractArray{State}`: An array representing the state of spins in the Ising graph.
- `ig::IsingGraph`: The Ising graph defining the interactions and biases.

# Returns:
- `Vector{Float64}`: An array of energy values for each state.
"""
function energy(σ::AbstractArray{<:State}, ig::IsingGraph)
    J, h = couplings(ig), biases(ig)
    dot.(σ, Ref(J), σ) + dot.(Ref(h), σ)
end

"""
$(TYPEDSIGNATURES)

Calculates the energy of a state in an Ising graph.

This function computes the energy of a given state in the context of an Ising graph. 
The energy is calculated based on the interactions between spins and their associated biases.

# Arguments:
- `ig::IsingGraph{T}`: The Ising graph defining the interactions and biases.
- `ig_state::Dict{Int, Int}`: A dictionary mapping spin indices to their corresponding states.

# Returns:
- `T`: The energy of the state in the Ising graph.
"""
function energy(ig::IsingGraph{T}, ig_state::Dict{Int,Int}) where {T}
    en = zero(T)
    for (i, σ) ∈ ig_state
        en += get_prop(ig, i, :h) * σ
        for (j, η) ∈ ig_state
            if has_edge(ig, i, j)
                en += T(1 / 2) * σ * get_prop(ig, i, j, :J) * η
            elseif has_edge(ig, j, i)
                en += T(1 / 2) * σ * get_prop(ig, j, i, :J) * η
            end
        end
    end
    en
end

"""
$(TYPEDSIGNATURES)

Generates the energy spectrum for an Ising graph.

This function computes the energy spectrum (energies and corresponding states) for a given Ising graph. 
The energy spectrum represents all possible energy levels and their associated states in the Ising graph.
    
# Arguments:
- `ig::IsingGraph{T}`: The Ising graph for which the energy spectrum is generated.
    
# Returns:
- `Spectrum`: An instance of the `Spectrum` type containing the energy levels and states.    
"""
function Spectrum(ig::IsingGraph{T}) where {T}
    L = nv(ig)
    N = 2^L
    energies = zeros(T, N)
    states = Vector{State}(undef, N)

    J, h = couplings(ig), biases(ig)
    Threads.@threads for i = 0:N-1
        σ = 2 .* digits(i, base = 2, pad = L) .- 1
        @inbounds energies[i+1] = dot(σ, J, σ) + dot(h, σ)
        @inbounds states[i+1] = σ
    end
    Spectrum(energies, states)
end

"""
$(TYPEDSIGNATURES)

Computes the Gibbs tensor for an Ising graph at a given inverse temperature.

This function calculates the Gibbs tensor for an Ising graph at a specified inverse temperature (β). 
The Gibbs tensor represents the conditional probabilities of states given the inverse temperature and the Ising graph.
    
# Arguments:
- `ig::IsingGraph{T}`: The Ising graph for which the Gibbs tensor is computed.
- `β::T (optional)`: The inverse temperature parameter. Default is 1.
    
# Returns:
- `Matrix{T}`: A matrix representing the Gibbs tensor with conditional probabilities.    
"""
function gibbs_tensor(ig::IsingGraph{T}, β::T = 1) where {T}
    σ = collect.(all_states(rank_vec(ig)))
    ρ = exp.(-β .* energy(σ, ig))
    ρ ./ sum(ρ)
end

function brute_force(ig::IsingGraph, s::Symbol = :CPU; num_states::Int = 1)
    brute_force(ig, Val(s); num_states)
end

"""
$(TYPEDSIGNATURES)

TODO only one of brute_force and full_spectrum should remain

Performs brute-force calculation of the lowest-energy states and their energies for an Ising graph.

This function exhaustively computes the lowest-energy states and their corresponding energies for an Ising graph.
The calculation is done using brute-force enumeration, making it feasible only for small Ising graphs.

# Arguments:
- `ig::IsingGraph{T}`: The Ising graph for which the lowest-energy states are computed.
- `::Val{:CPU}`: A value indicating that the computation is performed on the CPU.
- `num_states::Int (optional)`: The maximum number of lowest-energy states to calculate. Default is 1.

# Returns:
- `Spectrum`: A `Spectrum` object containing the lowest-energy states and their energies.
"""
function brute_force(ig::IsingGraph{T}, ::Val{:CPU}; num_states::Int = 1) where {T}
    L = nv(ig)
    L == 0 && return Spectrum(zeros(T, 1), Vector{Vector{Int}}[], zeros(T, 1))
    sp = Spectrum(ig)
    num_states = min(num_states, prod(rank_vec(ig)))
    idx = partialsortperm(vec(sp.energies), 1:num_states)
    Spectrum(sp.energies[idx], sp.states[idx])
end

function full_spectrum(ig::IsingGraph{T}; num_states::Int = 1) where {T}
    nv(ig) == 0 && return Spectrum(zeros(T, 1), Vector{Vector{Int}}[], zeros(T, 1))
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))
    σ = collect.(all_states(ig_rank))
    energies = energy(σ, ig)
    Spectrum(energies[begin:num_states], σ[begin:num_states])
end

function inter_cluster_energy(
    cl1_states::Vector{<:State},
    J::Matrix{<:Real},
    cl2_states::Vector{<:State},
)
    hcat(collect.(cl1_states)...)' * J * hcat(collect.(cl2_states)...)
end
