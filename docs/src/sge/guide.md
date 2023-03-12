# Introduction

## Example

```@example
using SpinGlassEngine, SpinGlassNetworks

# Prepare instance. Details can be found in SpinGlassNetworks.
instance = Dict((1, 1) => 0.5, (2, 2) => 0.75, (3, 3) => -0.25, (1, 2) => -1.0, (2, 3) => 1.0)
ig = ising_graph(instance)
fg = factor_graph(
    ig,
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice((4, 1, 1))
)

```