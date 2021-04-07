```@meta
Author = "Krzysztof Domino, Bartłomiej Gardas, Konrad Jałowiecki, Łukasz Pawela, Marek Rams, Anna Dziubyna"
```

# Welcome to SpinGlassPEPS documentation!
## Home
`SpinGlassPEPS` is an open-source Julia package for numerical computation in quantum information theory. 

!!! info "Star us on GitHub!" 
    If you have found this library useful, please consider starring the GitHub repository. This gives us an accurate lower bound of the satisfied users.


## Getting started
In this section we will provide a condensed overview of the package.

`SpinGlassPEPS.jl` is a collection of Julia packages bundled together under a single package `SpinGlassPEPS`. It can be installed using the Julia package manager for Julia v1.5 and higher. Inside the Julia REPL, type ] to enter the Pkg REPL mode and then run
```julia
using Pkg; 
Pkg.add("SpinGlassPEPS")
```
The package `SpinGlassPEPS` includes:
* `SpinGlassTensors.jl` - contains auxiliary functions used in `SpinGlassPEPS`
* `SpinGlassNetworks.jl` - creates factor graph and Ising spin-glass model
* `SpinGlassEngine.jl` - search for low energy spectrum using PEPS and MPS


## Quick example
Let us consider optimization problem defined on a pathological instance defined as follows.

We can find a ground state of this instance using SpinGlassPEPS interface via the procedure below.

```jldoctest
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS, MetaGraphs

m = 3
n = 4
t = 3

β = 1.

L = n * m * t
num_states = 22

control_params = Dict(
    "bond_dim" => typemax(Int),
    "var_tol" => 1E-8,
    "sweeps" => 4.
)

instance = "/Users/annamaria/Documents/GitHub/SpinGlassPEPS.jl/test/instances/test_$(m)_$(n)_$(t).txt"

ig = SpinGlassNetworks.ising_graph(instance)

fg = SpinGlassNetworks.factor_graph(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

#for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
peps = SpinGlassEngine.PEPSNetwork(m, n, fg, β, :NW, control_params)

    # solve the problem using B & B
sol = SpinGlassEngine.low_energy_spectrum(peps, num_states)
#end
(x->round.(x, digits = 1)).(sol.energies)[1]
# output
-16.4
```

## Our goals

`SpinGlassPEPS.jl` was created to heuristically solve Ising-type optimization problems defined on quasi-2d lattices.
It enables to compute conditional probabilities and find the most probable states according to Gibbs distribution by contracting tensor networks. It is a powerful tool to reconstruct the low-energy spectrum of the model. 

We aim to provide fast, reliable and easy to use emulator of D-Wave ``2000``Q quantum annealers. Our solver calculates ``L \ll 2 ^N`` low energy states (and their corresponding energies) for ``N \le 2048``. 

## Citing SpinGlassPEPS
If you use `SpinGlassPEPS` for academic research and wish to cite it, please use the following paper:

K. Jałowiecki, K. Domino, A. M. Dziubyna, M. M. Rams, B. Gardas and Ł. Pawela, *“SpinGlassPEPS.jl: software to emulate quantum annealing processors”*