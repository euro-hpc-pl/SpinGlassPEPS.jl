# Brief description of the algorithm
We consider tensor network based algorithm for finding ground state configurations of quasi-2D Ising problems. We employ tensor networks to represent the Gibbs distribution [1]. Then we use approximate tensor network contraction to efficiently identify the low-energy spectrum of some quasi-two-dimensional Hamiltonians [2].

Let us consider a classical Ising Hamiltonian
```math
H(\underline{s}_N) =  \sum_{\langle i, j\rangle \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i =1}^N J_{ii} s_i
```
where $\underline{s}_N$ denotes a particular configuration of $N$ binary variables $s_i=\pm 1$. More generally, we  mark sub-configurations of the first $k$ variables as $\underline{s}_k = (s_1, s_2, \ldots, s_k)$. Non-zero couplings $J_{ij} \in \mathbb{R}$ are input parameters of a given problem instance and form a connectivity graph $\mathcal{E}$.

## Graphs with large unit cells
We assume that graph $\mathcal{E}$ forms a quasi-2D lattice. In real life applications such graphs have large unit cells approaching 24 spins. `SpinGlassPEPS.jl` allows for unit cells containing multiple spins. 

!!! info
    More information on lattice geometries you can find in section Lattice Geometries (TODO: link).

```@raw html
<img src="../images/lattice.png" width="200%" class="center"/>
```
In order to adress this three types of geometries using tensor networks, we represent the problem as a clustered Hamiltonian. To that end we group together sets of variables. In this framework Ising problem translates to:
```math
H(\underline{x}_{\bar{N}}) = \sum_{\langle m,n\rangle \in \mathcal{F}} E_{x_m x_n} + \sum_{n=1}^{\bar{N}} E_{x_n}
```
where $\mathcal{F}$ forms a 2D graph, in which we indicate nearest-neighbour interactions with blue lines, and diagonal connections with green lines in the picture above.
Each $x_n$ takes $d$ values with  $d=2^4$ for square diagonal, $d=2^{24}$ for Pegasus and $2^{16}$ for Zephyr geometry. 
$E_{x_n}$ is an intra-node energy of the corresponding binary-variables configuration, and $E_{x_n x_m}$ is inter-node energy.
```@raw html
<img src="../images/clustering.png" width="200%" class="center"/>
```
## Calculating conditional probabilities
We assume that finding low energy states is equivalent to finding most probable states.
We represent the probability distribution as a PEPS tensor network.
```math
    p(\underline{x}_{\bar{N}}) = \frac{1}{Z} \exp{(-\beta H(\underline{x}_{\bar{N}}))}
```
where $Z$ is a partition function and $\beta$ is inverse temperature. Once the PEPS tensor network is constructed, the probability distribution can be obtained by approximately contracting the tensor network, which is described in more details in subsection Tensor network contractions for optimization problems (TODO: link).
Subsequently, we select only the configurations with the highest marginal probabilities
```math
    p(\underline{x}_{n+1}) = p(x_{n+1} | \underline{x}_{n}) \times p(\underline{x}_{n})
```

## Branch and bound search
By employing branch and bound search strategy iteratively row after row, we address the solution of Hamiltonian in the terms of conditional probabilities. This approach enables the identification of most probable (low-energy) spin configurations within the problem space. 

```@raw html
<img src="../images/bb.png" width="200%" class="center"/>
```

## Tensor network contractions for optimization problems
Branch and bound search relies on the calculation of conditional probabilities. To that end, we use tensor network techniques. Conditional probabilities are obtained by contracting a PEPS tensor network, which, although an NP-hard problem, can be computed approximately. The approach utilized is boundary MPS-MPO, which involves contracting a tensor network row by row and truncating the bond dimension.

```@raw html
<img src="../images/prob.png" width="150%" class="center"/>
```
```@raw html
<img src="../images/explain.png" width="75%" class="center"/>
```

## References & Related works

1. "Two-Dimensional Tensor Product Variational Formulation" T. Nishino, Y. Hieida, K. Okunishi, N. Maeshima, Y. Akutsu, A. Gendiar, [Progr. Theor. Phys. 105, 409 (2001)](https://academic.oup.com/ptp/article/105/3/409/1834124)

2. "Approximate optimization, sampling, and spin-glass droplet discovery with tensor networks" Marek M. Rams, Masoud Mohseni, Daniel Eppens, Konrad Jałowiecki, Bartłomiej Gardas [Phys. Rev. E 104, 025308 (2021)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.025308) or arXiv version [arXiv:1811.06518](https://arxiv.org/abs/1811.06518)

3. [tnac4o](https://github.com/marekrams/tnac4o/tree/master)