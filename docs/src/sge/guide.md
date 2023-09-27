# Introduction
We consider a classical Ising Hamiltonian
```math
E = \sum_{<i,j> \in \mathcal{E}} J_{ij} s_i s_j + \sum_j h_i s_j.
```
where ``s`` is a configuration of ``N`` classical spins taking values ``s_i = \pm 1``
and ``J_{ij}, h_i \in \mathbb{R}`` are input parameters of a given problem instance. 
Nonzero couplings ``J_{ij}`` form a graph ``\mathcal{E}``. Edges of ``\mathcal{E}`` form a quasi-two-dimensional structure. In this package we focus in particular on the [Chimera](https://docs.dwavesys.com/docs/latest/c_gs_4.html#chimera-graph) graph with up to 2048 spins. 


## Finding structure of low energy states
Below we describe simple Ising chain spin system with open boundary condition. The system has three spins with couplings ``J_{12} = -1.0`` and``J_{23} = 1.0``. Additionaly there are local fields ``h_1 = 0.5``, ``h_2 = 0.75`` and ``h_3 = -0.25``. 

We can calculate spectrum using `SpinGlassPEPS`. First we create graph (called Ising graph) which corespond to given Ising system. Then from this graph we create PEPS tensor network. Lastly we define model's parameters and control parameters such as `num_states` - maximal number of low energy states to be found. Then we can use function `low_energy_spectrum` to find desired low energy spectrum.


