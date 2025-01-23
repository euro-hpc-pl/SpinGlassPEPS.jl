## Belief propagation
The `SpinGlassPEPS.jl` package is capable of handling clusters with up to 24 spins, which results in a total of 2^24 degrees of freedom per cluster. This makes the contraction of the tensor network generated from such a Hamiltonian computationally expensive. To address this, `SpinGlassPEPS.jl` offers an optional feature for local dimensional reduction of cluster degrees of freedom by selectively choosing the most probable states within each cluster. This method reduces the dimensionality of the problem by focusing on the most relevant and energetically favorable states. The marginal probabilities of each Potts variable are approximated using the Loopy Belief Propagation (LBP) algorithm.

```@docs
potts_hamiltonian_2site
belief_propagation
truncate_potts_hamiltonian_2site_BP
```