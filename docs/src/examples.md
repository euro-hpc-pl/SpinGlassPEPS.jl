# Introduction
In this section, we will introduce two illustrative examples to highlight the effectiveness of the software. The first example will delve into a classical Ising problem commonly encountered in research, while the second will focus on an optimization problem related to Random Markov Fields. These examples serve as practical demonstrations of `SpinGlassPEPS.jl` in action, providing valuable insights into the software's capabilities and showcasing its utility across a wide range of scenarios.
# Ising type optimization problems
Ising type optimization problems with the cost function:

$$H =  \sum_{(i,j) \in \mathcal{E}} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

where $J_{ij}$ is the coupling constant between spins $i$ and $j$, $s_i$, $s_j$ can take on the values of either +1 or -1, $h_i$ is the external magnetic field at spin $i$, and the sum is taken over all pairs of spins and all spins in the system $\mathcal{E}$, respectively.

## Ground state search on super square lattice

In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a square lattice with next nearest neighbors interactions on 100 spins.
```@raw html
<img src="../images/square_diag.pdf" width="70%" class="center"/>
```
The package is used to explore various strategies for solving the problem, and it provides functionalities for performing Hamiltonian clustering, belief propagation, and low-energy spectrum searches using different MPS (Matrix Product State) strategies.

First, we set up the problem by defining the lattice and specifying various parameters such as temperature (β), bond dimension, and search parameters. We also create a clustered Hamiltonian using the specified lattice and perform clustering calculations.

Next, we select the MPS strategy (in this case, Zipper) and other parameters for the network contractor. We create a PEPS (Projected Entangled Pair State) network and initialize the contractor with this network, along with the specified parameters.

Finally, we perform a low-energy spectrum search using the initialized contractor, exploring different branches of the search tree. The example showcases how SpinGlassPEPS can be utilized to find the lowest energy configurations for a Spin Glass system.


```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

function bench(instance::String)
    m, n, t = 5, 5, 4
    onGPU = true

    β = 1.0
    bond_dim = 12
    δp = 1E-4
    num_states = 20

    ig = ising_graph(instance)
    cl_h = clustered_hamiltonian(
        ig,
        spectrum = full_spectrum,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )
    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)
    Gauge = NoUpdate
    graduate_truncation = :graduate_truncate
    for Strategy ∈ (SVDTruncate, MPSAnnealing, Zipper), transform ∈ all_lattice_transformations
        for Layout ∈ (GaugesEnergy, EnergyGauges, EngGaugesEng), Sparsity ∈ (Dense, Sparse)
            net = PEPSNetwork{SquareCrossSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
            ctr = MpsContractor{Strategy, Gauge}(net, [β/8, β/4, β/2, β], graduate_truncation, params; onGPU=onGPU)
            # sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
            clear_memoize_cache()
        end
    end
end

bench("$(@__DIR__)/../src/instances/square_diagonal/5x5/diagonal.txt")
```
## Ground state search on Pegasus lattice
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a D-Wave Pegasus lattice with 216 spins and 1324 couplings.
```@raw html
<img src="../images/pegasus.pdf" width="70%" class="center"/>
```

```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
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

ig = ising_graph("$(@__DIR__)/../src/instances/pegasus_random/P4/RAU/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

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

## Droplet search on Pegasus lattice
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a D-Wave Pegasus lattice with 216 spins and 1324 couplings.


```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
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
energy_cutoff = 1
hamming_cutoff = 20
VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2

ig = ising_graph("$(@__DIR__)/../src/instances/pegasus_random/P4/RAU/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

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

# sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDroplets(energy_cutoff, hamming_cutoff, :hamming)))
# println("Lowest energy found: ", sol.energies)

```


## Ground state search on Pegasus lattice with local dimensional reduction
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a D-Wave Pegasus lattice with 216 spins and 1324 couplings with local dimensional reduction based on Loopy Belief Propagation algorithm.

```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
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
new_cl_h = clustered_hamiltonian_2site(cl_h, β)
beliefs = belief_propagation(new_cl_h, β; tol=1e-6, iter=iter)
cl_h = truncate_clustered_hamiltonian_2site_BP(cl_h, beliefs, cs; beta=β)

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


## Ground state search on Zephyr lattice
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to perform a low-energy spectrum search for a Spin Glass Hamiltonian defined on a D-Wave Zephyr lattice with 332 spins and 2735 couplings.
```@raw html
<img src="../images/zephyr.pdf" width="70%" class="center"/>
```

```@example
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS
using SpinGlassExhaustive
using Logging
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m = 6
n = 6
t = 4

onGPU = true
β = 0.5
bond_dim = 5
DE = 16.0
δp = 1E-5*exp(-β * DE)
num_states = 128

VAR_TOL = 1E-16
MS = 0
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2

ig = ising_graph("$(@__DIR__)/../src/instances/zephyr_random/Z3/RAU/001_sg.txt")

cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=zephyr_lattice((m, n, t))
)

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

# Random Markov Field optimization problems
Random Markov Field type model on a 2D square lattice with cost function
$$H =  \sum_{(i,j) \in \mathcal{E}} E(s_i, s_j) + \sum_{i} E(s_i)$$
and nearest-neighbour interactions only.