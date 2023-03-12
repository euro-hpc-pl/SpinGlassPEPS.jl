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


```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS, MetaGraphs

# Create instance and coresponding factor graph. Details can be
# found in SpinGlassNetworks documentation.
instance = Dict((1, 1) => 0.5, (2, 2) => 0.75, (3, 3) => -0.25, (1, 2) => -1.0, (2, 3) => 1.0)
ig = ising_graph(instance)
fg = factor_graph(
        ig,
        cluster_assignment_rule = super_square_lattice((3, 1, 1)),
    )

# Define inverse temperature for gibbs distribution used in tensor network 
# contraction. Details can be found in SpinGlassEngine documentation
β = 1.0

# Create PEPS network
peps = PEPSNetwork(3, 1, fg, rotation(0), β = β, bond_dim = 32)

# Decide number of states we wane
num_states = 3

# Solve model
sol = low_energy_spectrum(peps, num_states)
@show sol.states, sol.energies

```
