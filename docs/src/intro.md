# Getting started
Before diving into the documentation for the provided functionalities, let's demonstrate the core capabilities of this package through a practical example.

## Basic example
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to calculate a low-energy spectrum for a spin glass Hamiltonian defined on a square lattice with diagonal interactions. 

```@julia
using SpinGlassEngine
using SpinGlassNetworks

function get_instance(topology::NTuple{3, Int})
    m, n, t = topology
    "$(@__DIR__)/instances/$(m)x$(n)x$(t).txt"
end

function run_square_diag_bench(::Type{T}; topology::NTuple{3, Int}) where {T}
    m, n, _ = topology
    instance = get_instance(topology)
    lattice = super_square_lattice(topology)

    hamming_dist = 5
    eng = 10

    best_energies = T[]

    potts_h = potts_hamiltonian(
        ising_graph(instance),
        spectrum = full_spectrum,
        cluster_assignment_rule = lattice,
    )

    params = MpsParameters{T}(; bond_dim = 16, num_sweeps = 1)
    search_params = SearchParameters(; max_states = 2^8, cutoff_prob = 1E-4)

    for transform ∈ all_lattice_transformations
        net = PEPSNetwork{KingSingleNode{GaugesEnergy}, Dense, T}(
            m, n, potts_h, transform,
        )

        ctr = MpsContractor(SVDTruncate, net, params; 
            onGPU = false, beta = T(2), graduate_truncation = true,
        )

        droplets = SingleLayerDroplets(; max_energy = 10, min_size = 5, metric = :hamming)
        merge_strategy = merge_branches(
            ctr; merge_prob = :none , droplets_encoding = droplets,
        )

        sol, _ = low_energy_spectrum(ctr, search_params, merge_strategy)
        sol2 = unpack_droplets(sol, T(2))
        ig_states = decode_potts_hamiltonian_state.(Ref(potts_h), sol2.states)
        ldrop = length(sol2.states)

        println("Number of droplets for transform $(transform) is $(ldrop)")
        println("Droplet energies: $(sol2.energies)")

        push!(best_energies, sol.energies[1])
        clear_memoize_cache()
    end

    ground = best_energies[1]
    @assert all(ground .≈ best_energies)

    println("Best energy found: $(ground)")
end

T = Float64
@time run_square_diag_bench(T; topology = (3, 3, 2))
```

### Main steps
Let’s walk through the key steps of the code.

#### Defining the lattice
The first line of the code above loads the problem instance using the `get_instance` function:
```@julia
instance = get_instance(topology)
```
Here, `topology` is a tuple `(m, n, t)` representing the dimensions of the lattice `m`, `n` and the cluster size `t`. The `get_instance` function constructs the file path to the corresponding problem instance, based on the provided topology.

The topology of the lattice is specified: 
```@julia
topology = (3, 3, 2)
m, n, _ = topology
```
This defines a 3x3 grid with clusters of size 2.

#### Defining the Hamiltonian
Then we map the Ising problem to a Potts Hamiltonian defined on a king’s graph (`super_square_lattice`):
```@julia
potts_h = potts_hamiltonian(
ising_graph(instance),
spectrum = full_spectrum,
cluster_assignment_rule = super_square_lattice(topology),
)
```
Here, `ising_graph(instance)` reads the Ising graph from the file and the spins are grouped into clusters based on the `super_square_lattice rule`.

#### Setting parameters
To control the complexity and accuracy of the simulation, we define several parameters:
```@julia
params = MpsParameters{T}(; bond_dim = 16, num_sweeps = 1)
search_params = SearchParameters(; max_states = 2^8, cut_off_prob = 1E-4)
```
* `bond_dim = 16`: The bond dimension for the tensor network.
* `num_sweeps = 1`: Number of sweeps during variational compression.
* `max_states = 2^8`: Maximum number of states considered during the search.
* `cut_off_prob = 1E-4`: The cutoff probability specifies the probability below which states are discarded from further consideration.

#### Tensor network construction and contraction
The tensor network representation of the system is created using the `PEPSNetwork` structure:
```@julia
net = PEPSNetwork{KingSingleNode{GaugesEnergy}, Dense, T}(m, n, potts_h, transform)
```
This constructs a PEPS network based on the `KingSingleNode`, which specifies the type of the node used within the tensor networks. The layout `GaugesEnergy` defines how the tensor network is divided into boundary Matrix Product States (MPSs). 
Other control parameter includes `Sparsity` which determines whether dense or sparse tensors should be used. In this example, as we apply small clusters containing two spins, we can use `Dense` mode. The parameter `T` represents the data type used for numerical calculations. In this example, we set: 
```@julia
T = Float64
```
Here, `Float64` specifies that the computations will be performed using 64-bit floating-point numbers, which is a common choice in scientific computing for balancing precision and performance. One can also use `Float32`.
The contraction of the tensor network is handled by the `MpsContractor`:
```@julia
ctr = MpsContractor(SVDTruncate, net, params; 
    onGPU = false, beta = T(2), graduate_truncation = true,
)
```
The parameters for the `MpsContractor` include:
* `Strategy` refers to the method used for approximating boundary Matrix Product States. Here `Strategy` is set to `SVDTruncate`.
* `onGPU = false`: Here computations are done on the CPU. If you want to switch on GPU mode, then type
```@julia
onGPU = true
``` 
* `beta = T(2)`: Inverse temperature, here set to 2. Higher value (lower temperature) allows us to focus on low-energy states.
* `graduate_truncation = true`: Enabling gradual truncation of the MPS.

#### Searching for excitations
The branch-and-bound search algorithm is used to find low-energy excitations:
```@julia
single = SingleLayerDroplets(eng, hamming_dist, :hamming)
merge_strategy = merge_branches(ctr; merge_type = :nofit, update_droplets = single)
```
Here, the parameter `eng` sets the energy range within which we look for excitations above the ground state. The `hamming_dist` parameter enforces a minimum Hamming distance between excitations, ensuring that the algorithm searches for distinct, independent excitations.
The optional `merge_branches` function allows us to identify spin glass droplets.

#### Multiple lattice transformations
We apply different lattice transformations to the problem, iterating over all possible transformations:
```@julia 
for transform ∈ all_lattice_transformations
    ...
end
```
This loop applies all possible transformations (rotations and reflections) to the 2D lattice. By exploring all eight transformations, the algorithm can start the contraction process from different points on the lattice, improving stability and increasing the chances of finding the global minimum energy state.

#### Low-energy spectrum calculation
Finally, the low-energy spectrum is calculated with:
```@julia
sol, _ = low_energy_spectrum(ctr, search_params, merge_strategy)
```
This function returns the low-energy states found during the search.

### Expected output
The output of this example should print the best energy found during the optimization:

```@julia
println("Best energy found: $(ground)")
```
This output confirms that the ground state energies found under different lattice transformations are consistent.

The full function is executed as:
```@julia
T = Float64
@time run_square_diag_bench(T; topology = (3, 3, 2))
```
