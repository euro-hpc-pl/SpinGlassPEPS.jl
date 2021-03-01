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

function brute_force(ig::MetaGraph; num_states::Int=1)
    if nv(ig) == 0 return Spectrum(zeros(1), []) end
    ig_rank = rank_vec(ig)
    num_states = min(num_states, prod(ig_rank))

    σ = collect.(all_states(ig_rank))
    energies = energy.(σ, Ref(ig))
    perm = partialsortperm(vec(energies), 1:num_states)
    Spectrum(energies[perm], σ[perm])
end

function full_spectrum(cl::Cluster; num_states::Int=prod(cl.rank))
    if isempty(cl.vertices) return Spectrum(zeros(1), []) end
    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    Spectrum(energies[1:num_states], σ[1:num_states])
end
