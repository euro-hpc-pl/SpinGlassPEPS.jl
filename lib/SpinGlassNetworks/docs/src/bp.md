## Local dimensional reduction of cluster degrees of freedom
The `SpinGlassPEPS.jl` package addresses the computational challenges posed by large unit cells, which result in a high number of degrees of freedom per cluster. Contracting tensor networks generated from such Hamiltonians can become numerically expensive. To mitigate this, the package provides an optional local dimensional reduction feature, which reduces the problem's complexity by focusing on the most probable states within each cluster.

This dimensionality reduction is achieved by approximating the marginal probabilities of Potts variables using the Loopy Belief Propagation (LBP) algorithm. LBP iteratively updates messages between clusters and edges to approximate the likelihood of configurations within each cluster. While exact convergence is guaranteed only for tree-like graphs, this method effectively selects a subset of energetically favorable states, even for geometries with loops, such as Pegasus and Zephyr lattices. The details of the Loopy Belief Propagation algorithm are described in [Ref.](https://arxiv.org/abs/2411.16431)

```@docs
truncate_potts_hamiltonian
potts_hamiltonian_2site
belief_propagation
truncate_potts_hamiltonian_2site_BP
```