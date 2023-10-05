# Examples
In this example, we demonstrate how to use the SpinGlassPEPS package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a Pegasus lattice. The package is used to explore various strategies for solving the problem, and it provides functionalities for performing Hamiltonian clustering, belief propagation, and low-energy spectrum searches using different MPS (Matrix Product State) strategies.

First, we set up the problem by defining the lattice and specifying various parameters such as temperature (β), bond dimension, and search parameters. We also create a clustered Hamiltonian using the specified lattice and perform clustering calculations.

Next, we select the MPS strategy (in this case, Zipper) and other parameters for the network contractor. We create a PEPS (Projected Entangled Pair State) network and initialize the contractor with this network, along with the specified parameters.

Finally, we perform a low-energy spectrum search using the initialized contractor, exploring different branches of the search tree. The example showcases how SpinGlassPEPS can be utilized to find the lowest energy configurations for a Spin Glass system.


```@example
using SpinGlassExhaustive
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 3
n = 3
t = 3

onGPU = true
β = 0.5
bond_dim = 8
DE = 16.0
δp = 1E-5*exp(-β * DE)
num_states = 128

VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
iter = 1
cs = 2^10
ig = ising_graph("$(@__DIR__)/../src/instances/pegasus_random/P4/RAU/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)
# new_cl_h = clustered_hamiltonian_2site(cl_h, β)
# beliefs = belief_propagation(new_cl_h, β; tol=1e-6, iter=iter)
# cl_h = truncate_clustered_hamiltonian_2site_BP(cl_h, beliefs, cs; beta=β)

params = MpsParameters(bond_dim, VAR_TOL, MS, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT)
search_params = SearchParameters(num_states, δp)

# Solve using PEPS search
energies = Vector{Float64}[]
Strategy = Zipper # MPSAnnealing # SVDTruncate
Sparsity = Sparse #Dense
Layout = GaugesEnergy
Gauge = NoUpdate

net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparsity}(m, n, cl_h, rotation(0))
ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)

# sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
# println("Lowest energy found: ", sol.energies)

```