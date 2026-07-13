# Introduction
The model-construction component uses a Potts Hamiltonian to transform Ising graphs into clustered representations. Instead of treating individual spins as separate variables, spins are grouped into clusters corresponding to unit cells of a given lattice geometry. This process reduces the number of variables while increasing their dimensionality, making the system more manageable for tensor-network-based approaches.

```@docs
potts_hamiltonian
```

## Simple example

```@example
using SpinGlassPEPS

# Load instance
instance = joinpath(
    pkgdir(SpinGlassPEPS), "docs", "src", "instances", "square_diagonal", "5x5", "diagonal.txt",
)
ig = ising_graph(instance)

# Create Potts Hamiltonian
potts_h = potts_hamiltonian(
    ig,
    cluster_assignment_rule = super_square_lattice((5,5,4))
)
```
