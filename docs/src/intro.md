# Getting started
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Introduction
We consider tensor network based algorithm for finding ground state configurations of quasi-2D problems. We employ tensor networks to represent the Gibbs distribution. Then we use approximate tensor network contraction to efficiently identify the low-energy spectrum of some quasi-two-dimensional Hamiltonians.

### Ising spin glass problems
Let us consider a classical Ising Hamiltonian
```math
H(\underline{s}_N) =  \sum_{\langle i, j\rangle \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i =1}^N J_{ii} s_i,
```
where ``\underline{s}_N`` denotes a particular configuration of ``N`` binary variables ``s_i=\pm 1``. Non-zero couplings ``J_{ij} \in \mathbb{R}`` are input parameters of a given problem instance and form a connectivity graph ``\mathcal{E}``.

Inspired by near-term quantum annealers, we assume that graph ``\mathcal{E}`` forms a quasi-2D lattice — in particular on a [Pegasus](https://docs.dwavesys.com/docs/latest/c_gs_4.html#pegasus-graph) and [Zephyr](https://docs.dwavesys.com/docs/latest/c_gs_4.html#zephyr-graph) graphs (D-Wave). Unit cells for Pegasus and Zephyr graphs consist of 24 and 16 spins respectively. Our solver is, however, more generall than D-Wave graphs. It can search for solutions of problems on square diagonal lattices. 

`SpinGlassPEPS.jl` translates the problem into a tensor network language, and performing branch and bound search, finds the low energy spectrum of a given problem.

```@raw html
<img src="../images/peps_graph.pdf" width="200%" class="center"/>
```
### Random Markov Fields
Random Markov Field type model on a 2D square lattice with cost function
$$H =  \sum_{(i,j) \in \mathcal{E}} E(s_i, s_j) + \sum_{i} E(s_i)$$
and nearest-neighbour interactions only.

## Branch and bound search
By employing branch and bound search strategy iteratively row after row, we address the solution of Hamiltonian in the terms of conditional probabilities. This approach enables the identification of most probable (low-energy) spin configurations within the problem space. 

```@raw html
<img src="../images/bb.pdf" width="200%" class="center"/>
```

## Calculating conditional probabilities

In order to indentify most probable states we need to calculate the conditional probabilities. Conditional probabilities are obtained by contracting a PEPS tensor network, which, although an NP-hard problem, can be computed approximately. The approach utilized is boundary MPS-MPO, which involves contracting a tensor network row by row and truncating the bond dimension.

```@raw html
<img src="../images/prob.pdf" width="150%" class="center"/>
```