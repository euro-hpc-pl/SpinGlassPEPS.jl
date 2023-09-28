# Introduction
We consider a classical Ising Hamiltonian
```math
E = \sum_{<i,j> \in \mathcal{E}} J_{ij} s_i s_j + \sum_j h_i s_j.
```
where ``s`` is a configuration of ``N`` classical spins taking values ``s_i = \pm 1``
and ``J_{ij}, h_i \in \mathbb{R}`` are input parameters of a given problem instance. 
Nonzero couplings ``J_{ij}`` form a graph ``\mathcal{E}``. Edges of ``\mathcal{E}`` form a quasi-two-dimensional structure. In this package we focus in particular on the [Pegasus](https://docs.dwavesys.com/docs/latest/c_gs_4.html#pegasus-graph) and [Zephyr](https://docs.dwavesys.com/docs/latest/c_gs_4.html#zephyr-graph) graphs.


## Finding structure of low energy states

