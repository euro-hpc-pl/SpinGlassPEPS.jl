# Brief description of the algorithm
We consider tensor network based algorithm for finding ground state configurations of quasi-2D Ising problems. We employ tensor networks to represent the Gibbs distribution [1]. Then we use approximate tensor network contraction to efficiently identify the low-energy spectrum of some quasi-two-dimensional Hamiltonians [2].

Let us consider a classical Ising Hamiltonian
```math
H(\underline{s}_N) =  \sum_{\langle i, j\rangle \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i =1}^N J_{ii} s_i
```
where $\underline{s}_N$ denotes a particular configuration of $N$ binary variables $s_i=\pm 1$. Non-zero couplings $J_{ij} \in \mathbb{R}$ are input parameters of a given problem instance and form a connectivity graph $\mathcal{E}$.

## Graphs with large unit cells
We assume that graph $\mathcal{E}$ forms a quasi-2D lattice. In real life applications such graphs have large unit cells approaching 24 spins. `SpinGlassPEPS.jl` allows for unit cells containing multiple spins. 

!!! info
    More information on lattice geometries you can find in section Lattice Geometries (TODO: link).

```@raw html
<img src="../images/lattice.pdf" width="200%" class="center"/>
```
Next step in the algorithm is to build clustered Hamiltonian, in which the Ising problem translates to:
```math
H(\underline{x}{\bar{N}}) = \sum_{\langle m,n\rangle \in \mathcal{F}} E_{x_m x_n} + \sum_{n=1}^{\bar{N}} E_{x_n}
```
$\mathcal{F}$ forms a 2D graph, where we indicate nearest-neighbour interactions with blue lines, and diagonal connections with green lines.
Each $x_n$ takes $d$ values with  $d=2^4$ for square diagonal, $d=2^{24}$ for Pegasus and $2^{16}$ for Zephyr geometry. 
$E_{x_n}$ is an intra-node energy of the corresponding binary-variables configuration, and $E_{x_n x_m}$ is inter-node energy.

After creating the clustered Hamiltonian, we can turn it into a PEPS tensor network as shown in the figure below. In all lattices we support, the essential components resembling unit cells are represented by red circles. Spins in adjacent clusters interacted with each other, which is depicted by blue squares. Additionally, we permit diagonal interactions between next nearest neighbors, mediated by green squares.

```@raw html
<img src="../images/pepstn.pdf" width="200%" class="center"/>
```

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

## References & Related works

1. "Two-Dimensional Tensor Product Variational Formulation" T. Nishino, Y. Hieida, K. Okunishi, N. Maeshima, Y. Akutsu, A. Gendiar, [Progr. Theor. Phys. 105, 409 (2001)](https://academic.oup.com/ptp/article/105/3/409/1834124)

2. "Approximate optimization, sampling, and spin-glass droplet discovery with tensor networks" Marek M. Rams, Masoud Mohseni, Daniel Eppens, Konrad Jałowiecki, Bartłomiej Gardas [Phys. Rev. E 104, 025308 (2021)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.025308) or arXiv version [arXiv:1811.06518](https://arxiv.org/abs/1811.06518)

3. [tnac4o](https://github.com/marekrams/tnac4o/tree/master)