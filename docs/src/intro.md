# Getting started
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Introduction
We consider tensor network based algorithm for finding ground state configurations of quasi-2D Ising problems. We employ tensor networks to represent the Gibbs distribution. Then we use approximate tensor network contraction to efficiently identify the low-energy spectrum of some quasi-two-dimensional Hamiltonians.

Let us consider a classical Ising Hamiltonian
```math
H(\underline{s}_N) =  \sum_{\langle i, j\rangle \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i =1}^N J_{ii} s_i,
```
where ``\underline{s}_N`` denotes a particular configuration of ``N`` binary variables ``s_i=\pm 1``. Non-zero couplings ``J_{ij} \in \mathbb{R}`` are input parameters of a given problem instance and form a connectivity graph ``\mathcal{E}``.

Inspired by near-term quantum annealers, we assume that graph ``\mathcal{E}`` forms a quasi-2D lattice — in particular on a [Pegasus](https://docs.dwavesys.com/docs/latest/c_gs_4.html#pegasus-graph) and [Zephyr](https://docs.dwavesys.com/docs/latest/c_gs_4.html#zephyr-graph) graphs (D-Wave). Unit cells for Pegasus and Zephyr graphs consist of 24 and 16 spins respectively.

```@raw html
<img src="../images/graphs.pdf" width="200%" class="center"/>
```

## Branch and bound search
By employing branch and bound search strategy iteratively row after row, we address the solution of Hamiltonian in the terms of conditional probabilities. This approach enables the identification of most probable (low-energy) spin configurations within the problem space. 

```@raw html
<img src="../images/bb.pdf" width="200%" class="center"/>
```

## Calculating conditional probabilities

```@raw html
<img src="../images/prob.pdf" width="150%" class="center"/>
```

```@raw html
<img src="../images/PEPS.pdf" width="120%" class="center"/>
```


## Finding structure of low energy states
Below we describe simple Ising chain spin system with open boundary condition. The system has three spins with couplings ``J_{12} = -1.0`` and``J_{23} = 1.0``. Additionaly there are local fields ``h_1 = 0.5``, ``h_2 = 0.75`` and ``h_3 = -0.25``. 

We can calculate spectrum using `SpinGlassPEPS`. First we create graph (called Ising graph) which corespond to given Ising system. Then from this graph we create PEPS tensor network. Lastly we define model's parameters and control parameters such as `num_states` - maximal number of low energy states to be found. Then we can use function `low_energy_spectrum` to find desired low energy spectrum.
