# Example
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Introduction
We consider a classical Ising Hamiltonian
```math
E = \sum_{<i,j> \in \mathcal{E}} J_{ij} s_i s_j + \sum_j h_i s_j.
```
where ``s`` is a configuration of ``N`` classical spins taking values ``s_i = \pm 1``
and ``J_{ij}, h_i \in \mathbb{R}`` are input parameters of a given problem instance. 
Nonzero couplings ``J_{ij}`` form a graph ``\mathcal{E}``. Edges of ``\mathcal{E}`` form a quasi-two-dimensional structure. In this package we focus in particular on the chimera graph with up to 2048 spins. 


## Finding structure of low energy states
Below we describe the simplest possible system of two spins with couplings ``J_{12} = -1.0`` and fields ``h_1 = 0.5``, ``h_2 = 0.75``. Energy in Ising model can be calculated directly as:
```math
E = -1.0 \cdot s_1 \cdot s_2 + 0.5 \cdot s_1 + 0.75 \cdot s_2
```
In two-spin system we have four possible states: ``[-1, -1], [1, 1], [1, -1], [-1, 1]`` with energies ``-2.25, 0.25, 0.75, 1.25`` respectively.

We can calculate spectrum using `SpinGlassPEPS`. First we define model's parameters, grid and control parameters such as `num_states` - maximal number of low energy states to be found. Then we are ready to create Ising spin-glass model by`ising_graph` using grid defined before. 


```jldoctest
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS, MetaGraphs

([-2.25, 0.25, 0.75, 1.25], [[1, 1], [2, 2], [2, 1], [1, 2]])
# output
([-2.25, 0.25, 0.75, 1.25], [[1, 1], [2, 2], [2, 1], [1, 2]])
```
