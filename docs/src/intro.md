# Getting started
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Basic example
In this example, we demonstrate how to use the `SpinGlassPEPS.jl` package to obtain a low-energy spectrum for a spin glass Hamiltonian defined on a square lattice with diagonal interactions on 100 spins. Let's discuss the main steps of the code.

The first line
```@julia
instance = "$(@__DIR__)/../src/instances/square_diagonal/5x5/diagonal.txt"
```
reads instance that is provided in txt format.

Next line defines the problem size
```@julia
m, n, t = 5, 5, 4
```
In this example, we have number of columns and row, `m` and `n` respectively, equal 5. Parameter `t` tells how many spins are creating a cluster.

`SpinGlassPEPS.jl` enables to perform calculations not only on CPU, but also on GPU. If you want to switch on GPU mode, then type
```@julia
onGPU = true
```

The next part of the code contains parmeters which user should provide before starting the calculations.
The main parameter is temperature, given as the inverse of temperature.
```@julia
β = 1.0
```
A higher `β` lets us focus more on low-energy states, but it might make the numerical stability of tensor network contraction a bit shaky. Figuring out the best β depends on the problem, and we might need to try different values in experiments for various instances.

Subsequently, the user can input parameters that will be utilized in exploring the state space, such as the cutoff probability for terminating the search `δp` and the maximum number of states considered during the search (`num_states`).
```@julia
# Search parameters
δp = 0 # The cutoff probability for terminating the search
num_states = 20 # The maximum number of states to be considered during the search
```

Another group of parameters describes the method of contracting the network using the boundary MPS-MPO approach.
```@julia
bond_dim = 12 # Bond dimension
max_num_sweeps = 10 # Maximal number of sweeps during variational compression
tol_var = 1E-16 # The tolerance for the variational solver used in MPS optimization
tol_svd = 1E-16 # The tolerance used in singular value decomposition (SVD)
iters_svd = 2 # The number of iterations to perform in SVD computations
iters_var = 1 # The number of iterations for variational optimization
dtemp_mult = 2 # A multiplier for the bond dimension
method = :psvd_sparse # The SVD method to use
```

We can also choose the arrangement of tensors forming the MPO (`Layout`) and the strategy to optimize boundary MPS (`Strategy`). The user also has the decision-making authority on whether the MPS will be truncated in a gradual manner (`graduate_truncation`). We can initiate our algorithm from various starting points. The parameter responsible for this choice is `transform`, which allows for the rotation and reflection of the tensor network, consequently altering the starting point for the exploration. Last, but not least, the user can also choose whether to use the `Sparse` or `Dense` mode. This choice should depend on the size of the unit cell in the specific problem at hand.
```@julia
Layout = GaugesEnergy # Way of decomposition of the network into MPO
Strategy = Zipper # Strategy to optimize MPS
graduate_truncation = :graduate_truncate # Gradually truncates MPS
transform = rotation(0) # Transformation of the lattice
Sparsity = Sparse # Use sparse mode, when tensors are large
```

The parameters provided by the user are then stored in data structures `MpsParameters` and `SearchParameters`.
```@julia
params = MpsParameters(bond_dim, tol_var, max_num_sweeps, 
                        tol_svd, iters_svd, iters_var, dtemp_mult, method)
search_params = SearchParameters(num_states, δp)
```

With this prepared set of parameters, we are ready for the actual computations. The first step is to create the Ising graph. In the Ising graph, nodes are formed by the spin positions, and interactions between them are represented by the edges.
```@julia
ig = ising_graph(instance)
```

Next, we need to translate our problem into a clustered problem, where several spins form an unit cell. This is achieved through the use of the `clustered_hamiltonian` function.
```@julia
cl_h = clustered_hamiltonian(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)
```

The next part of the code builds the PEPS tensor network compatible with previously defined clustered Hamiltonian.
```@julia
net = PEPSNetwork{SquareCrossSingleNode{Layout}, Sparsity}(m, n, cl_h, transform)
```

In order to start calculating conditional probabilities we need to define `MpsContractor` structure which stores the elements and information necessary for contracting the tensor network and, consequently, calculating the probability of a given configuration. 
```@julia
ctr = MpsContractor{Strategy, NoUpdate}(net, [β], graduate_truncation, params; onGPU=onGPU)
```

Finally the call
```@julia
sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
```
which runs branch and bound algorithm included in `SpinGlassPEPS.jl` It is actual solver, which iteratively explores the state space in search of the most probable states. The probabilities of a given configuration are calculated approximately through the contractions of the tensor network.

## Expected output
The function `low_energy_spectrum`, as its output, provides a wealth of information, which we will briefly discuss now.
```@julia
sol_peps, schmidt_val = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
```
It returns a Solution-type structure (`sol_peps`) and Schmidt values `schmidt_val`. In the `Solution` structure, the following information is recorded:
* energies - a vector containing the energies of the discovered states
If you want to display it, you can type:
```@julia
println(sol_peps.energies)
```
In this case, you should obtain:
```@julia
[-215.23679958927175]
```
* states - a vector of cluster state configurations corresponding to the energies
```@julia
println(sol_peps.states)
```
```@julia
println([[4, 4, 2, 16, 9, 8, 8, 2, 3, 8, 7, 1, 4, 10, 9, 2, 11, 2, 1, 2, 11, 3, 11, 8, 3]])
```
* probabilities - the probabilities associated with each discovered state
```@julia
println(sol_peps.probabilities)
```
```@julia
[-5.637640487579043]
```
* degeneracy
```@julia
println(sol_peps.degeneracy)
```
```@julia
[2]
```
* largest discarded probability - largest probability below which states are considered to be discarded
```@julia
println(sol_peps.largest_discarded_probability)
```
```@julia
-4.70579014404923
```
* Schmidt values - a dictionary containing Schmidt spectra for each MPS
```@julia
println(schmidt_val)
```
which are
```@julia
Dict{Any, Any}(5 => Any[1.0, 1.0, 0.0035931343796194565, 0.0015050888555964259, 0.0007184752751868924, 5.2741877514519126e-5, 4.137035816131772e-5, 0.00040017490729592366, 0.00021874495320028077, 6.827849766898342e-5], 4 => Any[1.0, 1.0, 1.704951518711975e-5, 0.00037890798675182353, 0.00011310427642297989, 0.001014257142680146, 0.0012672631937840461, 0.0005487312667512858, 0.0006741839581781018, 0.00012017531445170455], 6 => Any[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2 => Any[1.0, 1.0, 0.0001405854472578707, 0.0012280075890260514, 0.001177462193268373, 0.0029570115655969827, 0.002997829968910592, 0.0011163442379909382, 0.0010056280784881478, 0.00026431187613365595], 3 => Any[1.0, 1.0, 1.864183962070951e-5, 0.006059161388679921, 0.006793028602573968, 0.012337242616802302, 0.011721497080177857, 0.013791830543357657, 0.020430181282353188, 0.014653186648427675])
```
* statistics - a possible warning sign, with values ranging from [0, 2]. A nonzero value signifies that certain conditional probabilities derived from tensor network contraction were negative, suggesting a lack of numerical stability during contraction. Here we display the worst-case scenario.
```@julia
println(minimum(values(ctr.statistics)))
```
The output should give you
```@julia
0.0
```