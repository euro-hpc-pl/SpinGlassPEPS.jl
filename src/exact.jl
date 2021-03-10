export gibbs_tensor
export brute_force, full_spectrum

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
    states = collect.(all_states(rank_vec(ig)))
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

function brute_force(ig::MetaGraph; sorted=true, num_states::Int=1)
    if nv(ig) == 0 return Spectrum(zeros(1), []) end
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))

    σ = collect.(all_states(ig_rank))
    energies = energy.(σ, Ref(ig))
    if sorted
        perm = partialsortperm(vec(energies), 1:num_states)
        return Spectrum(energies[perm], σ[perm])
    else
        return Spectrum(energies[1:num_states], σ[1:num_states])
    end
end

full_spectrum(ig::MetaGraph; num_states::Int=1) = brute_force(ig, sorted=false, num_states=num_states)
