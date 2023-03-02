## Quick example
Let us consider optimization problem defined on a pathological instance defined as follows.

![summary_time](../assets/images/pathological.png)

```@raw html
<img src="../assets/images/pathological.png" width="25%"/>
``` 

```{execute="false"}
Pkg.add("GraphPlot")
```

```julia
using GraphPlot
```

```julia
using LightGraphs 
using PlotRecipes
using Colors


# construct a simple undirected graph
g = SimpleGraph(36)
add_edge!(g, 1, 4)
add_edge!(g, 1, 5)
add_edge!(g, 1, 6)
add_edge!(g, 4, 7)
add_edge!(g, 4, 6)
add_edge!(g, 5, 16)
add_edge!(g, 6, 16)
add_edge!(g, 5, 7)
add_edge!(g, 6, 8)
add_edge!(g, 6, 9)
add_edge!(g, 6, 18)
add_edge!(g, 16, 18)
add_edge!(g, 1, 13)
add_edge!(g, 13, 16)
add_edge!(g, 13, 18)
add_edge!(g, 18, 28)
add_edge!(g, 28, 29)
add_edge!(g, 29, 30)
add_edge!(g, 28, 30)
add_edge!(g, 28, 31)
add_edge!(g, 28, 32)
add_edge!(g, 28, 33)
add_edge!(g, 29, 31)
add_edge!(g, 29, 32)
add_edge!(g, 29, 33)
add_edge!(g, 30, 31)
add_edge!(g, 30, 32)
add_edge!(g, 30, 33)

nodelabel = collect(1:36)
#num_vertices(g)
membership = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10, 10, 10, 11, 11,11,12,12,12]
#nodecolor = [colorant"lightseagreen", colorant"orange"]
nodecolor = [RGBA(0.0,0.8,0.8,i) for i in 1:12]
# membership color
nodefillc = nodecolor[membership]
gplot(g, nodefillc=nodefillc, nodelabel=nodelabel)
# just plot it
#gplot(g)

```

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