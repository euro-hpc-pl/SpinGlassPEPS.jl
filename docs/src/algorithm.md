# Brief description of the algorithm
We consider tensor network based algorithm for finding ground state configurations of quasi-2D Ising problems. We employ tensor networks to represent the Gibbs distribution (TODO: \cite{Nishino}). Then we use approximate tensor network contraction to efficiently identify the low-energy spectrum of some quasi-two-dimensional Hamiltonians (TODO: cite).

Let us consider a classical Ising Hamiltonian
```math
H(\underline{s}_N) =  \sum_{\langle i, j\rangle \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i =1}^N J_{ii} s_i,
```
where $\underline{s}_N$ denotes a particular configuration of $N$ binary variables $s_i=\pm 1$. Non-zero couplings $J_{ij} \in \mathbb{R}$ form a connectivity graph $\mathcal{E}$.

## Graphs with large unit cells
We assume that graph $\mathcal{E}$ forms a quasi-2D lattice. In real life applications such graphs have large unit cells. `SpinGlassPEPS.jl` allows for unit cells containing multiple spins, even reaching dozens of spins. More information on lattice geometries you can find in section Lattice Geometries (TODO: link).

```@raw html
<img src="../images/peps_graph.pdf" width="200%" class="center"/>
```
Next step is to build clustered Hamiltonian, in which the Ising problem translates to:
```math
H(\conf{x}{\Nbar}) = \sum_{\langle m,n\rangle \in \mathcal{F}} E_{x_m x_n} + \sum_{n=1}^{\Nbar} E_{x_n}
```

$\mathcal{F}$ forms a 2D graph, where we indicate nearest-neighbour interactions with blue lines, and diagonal connections with green lines.
Each $x_n$ takes $d$ values with  $d=2^4$ for square diagonal, $d=2^{24}$ for Pegasus and $2^{16}$ for Zephyr geometry. 
$E_{x_n}$ is an intra--node energy of the corresponding binary-variables configuration, and $E_{x_n x_m}$ is inter--node energy.

After creating the clustered Hamiltonian, we can turn it into a PEPS tensor network as shown in the figure above. In all lattices we support, the essential components resembling unit cells are represented by red circles and described using the formula ``\underset{x_m}{\Tr} (\exp{(-\beta E_{x_m})} P^{ml}P^{mr}P^{mu}P^{md})``. Here, ``x_m`` refers to a group of spins within the cluster in focus, while ``P_{ml}, P_{mr}, \ldots`` denote subsets of spins within ``x_m`` interacting with neighboring clusters in various directions like left and right. Spins in adjacent clusters influence each other, depicted by blue squares. Additionally, we permit diagonal interactions between next nearest neighbors, facilitated by green squares.

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