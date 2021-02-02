export gibbs_tensor, brute_force, full_spectrum

"""
$(TYPEDSIGNATURES)

Calculates Gibbs state of a classical Ising Hamiltonian

# Details

Calculates matrix elements (probabilities) of \$\\rho\$
```math
\$\\bra{\\σ}\\rho\\ket{\\sigma}\$
```
for all possible configurations \$\\σ\$.
"""
function gibbs_tensor(ig::MetaGraph, β=Float64=1.0)
    rank = get_prop(ig, :rank)
    states = collect.(all_states(rank))
    ρ = exp.(-β .* energy.(states, Ref(ig)))
    ρ ./ sum(ρ)
end

"""
$(TYPEDSIGNATURES)

Return the low energy spectrum

# Details

Calculates \$k\$ lowest energy states
together with the coresponding energies
of a classical Ising Hamiltonian
"""
function brute_force(ig::MetaGraph; num_states::Int=1)
    cl = Cluster(ig, 0)
    brute_force(cl, num_states=num_states)
end

function brute_force(cl::Cluster; num_states::Int=1)
    if isempty(cl.vertices)
        return Spectrum(zeros(1), [])   
    end

    if num_states > prod(cl.rank) num_states = prod(cl.rank) end

    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    perm = partialsortperm(vec(energies), 1:num_states) 
    Spectrum(energies[perm], σ[perm])
end

function full_spectrum(cl::Cluster; num_states::Int=prod(cl.rank))
    if isempty(cl.vertices)
        return Spectrum(zeros(1), [])   
    end
    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    Spectrum(energies[1:num_states], σ[1:num_states])   
end