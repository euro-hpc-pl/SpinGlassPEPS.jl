# Examples
Before providing the documentation of the offered functionality, it is good to demonstrate exactly what the package does.

## Introduction
We consider a classical Ising Hamiltonian
```math
E = -\sum_{<i,j> \in \mathcal{E}} J_{ij} s_i s_j - \sum_j h_i s_j.
```
where ``s`` is a configuration of ``N`` classical spins taking values ``s_i = \pm 1``
and ``J_{ij}, h_i \in \mathbb{R}`` are input parameters of a given problem instance. 
Nonzero couplings ``J_{ij}`` form a graph ``\mathcal{E}``. 


## Finding structure of low energy states
Below we describe the simplest possible system of two spins with couplings ``J_{12} = -1.0`` and fields ``h_1 = 0.5``, ``h_2 = 0.75``. Energy in Ising model can be calculated directly as:
```math
E = -1.0 \cdot s_1 \cdot s_2 + 0.5 \cdot s_1 + 0.75 \cdot s_2
```
In two-spin system we have four possible states: ``[-1, -1], [1, 1], [1, -1], [-1, 1]`` with energies ``-2.25, 0.25, 0.75, 1.25`` respectively.

We can calculate it using `SpinGlassPEPS`. First we define model's parameters, grid and control parameters such as `num_states` - maximal number of low energy states to be found. Then we are ready to create `ising_graph` using grid defined before. 


```jldoctest
using SpinGlassEngine, SpinGlassTensors, SpinGlassNetworks, SpinGlassPEPS, MetaGraphs

 # Model's parameters
    J12 = -1.0
    h1 = 0.5
    h2 = 0.75

    # dict to be read
    D = Dict((1, 2) => J12,
             (1, 1) => h1,
             (2, 2) => h2,
    )
    # control parameters
    m, n = 1, 2
    L = 2
    β = 1.
    num_states = 4

    # read in pure Ising
    ig = SpinGlassNetworks.ising_graph(D)

    # construct factor graph with no approx
    fg = SpinGlassNetworks.factor_graph(
        ig,
        Dict(1 => 2, 2 => 2),
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => 1, 2 => 2), # treat it as a grid with 1 spin cells
    )

    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    # get BF results for comparison
    exact_spectrum = SpinGlassNetworks.brute_force(ig; num_states=num_states)
    ϱ = SpinGlassNetworks.gibbs_tensor(ig, β)

    # split on the bond
    p1, e, p2 = get_prop.(Ref(fg), 1, 2, (:pl, :en, :pr))
    
    en = [ J12 * σ * η for σ ∈ [-1, 1], η ∈ [-1, 1]]

    #for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
    peps = SpinGlassEngine.PEPSNetwork(m, n, fg, β, :NW, control_params)
    # solve the problem using B & B
    sol = SpinGlassEngine.low_energy_spectrum(peps, num_states)
    #end

    sol.energies, sol.states
# output
([-2.25, 0.25, 0.75, 1.25], [[1, 1], [2, 2], [2, 1], [1, 2]])
```
