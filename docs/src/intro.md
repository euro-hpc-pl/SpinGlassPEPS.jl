# Getting started
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

# Basic example
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to obtain a low-energy spectrum search for a spin glass Hamiltonian defined on a square lattice with diagonal interactions on 100 spins.

The package is used to explore various strategies for solving the problem, and it provides functionalities for performing Hamiltonian clustering, belief propagation, and low-energy spectrum searches using different MPS (Matrix Product State) strategies.

First, we set up the problem by defining the lattice and specifying various parameters such as temperature (β), bond dimension, and search parameters. We also create a clustered Hamiltonian using the specified lattice and perform clustering calculations.

Next, we select the MPS strategy (in this case, Zipper) and other parameters for the network contractor. We create a PEPS (Projected Entangled Pair State) network and initialize the contractor with this network, along with the specified parameters.

Finally, we perform a low-energy spectrum search using the initialized contractor, exploring different branches of the search tree. The example showcases how `SpinGlassPEPS.jl` can be utilized to find the lowest energy configurations for a spin glass system.


```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

instance = "$(@__DIR__)/../src/instances/square_diagonal/5x5/diagonal.txt"

# size of network, m - number of columns, n - number of rows, t - number of spins in cluster
m, n, t = 5, 5, 4

onGPU = true # calculations on GPU (true) or on CPU (false) 

β = 1.0 # inverse temperature

# Search parameters
δp = 0 # The cutoff probability for terminating the search
num_states = 20 # The maximum number of states to be considered during the search

# MpsParameters 
bond_dim = 12 # Bond dimension
max_num_sweeps = 10 # Maximal number of sweeps durion variational compression
tol_var = 1E-16 # The tolerance for the variational solver used in MPS optimization
tol_svd = 1E-16 # The tolerance used in singular value decomposition (SVD)
iters_svd = 2 # The number of iterations to perform in SVD computations
iters_var = 1 # The number of iterations for variational optimization
dtemp_mult = 2 # A multiplier for the bond dimension
method = :psvd_sparse # The SVD method to use

# Contraction parameters and methods 
graduate_truncation = :graduate_truncate # Gradually truncates MPS
Strategy = Zipper # Strategy to optimize MPS
transform = rotation(0) # Transformation of the lattice
Layout = GaugesEnergy # Way of decomposition of the network into MPS
Sparsity = Sparse # Use sparse mode, when tensors are sparse

# Create Ising graph
ig = ising_graph(instance)
# Create clustered Hamiltonian
cl_h = clustered_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)
# Store parameters in structures
params = MpsParameters(bond_dim, tol_var, max_num_sweeps, 
                        tol_svd, iters_svd, iters_var, dtemp_mult, method)
search_params = SearchParameters(num_states, δp)
# Build tensor network
net = PEPSNetwork{SquareCrossSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
ctr = MpsContractor{Strategy, NoUpdate}(net, [β], graduate_truncation, params; onGPU=onGPU)
# Solve using branch and bound search
# sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
```