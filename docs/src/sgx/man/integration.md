# Solver integration

Model construction, exhaustive search, and the PEPS solver now ship in the same package:

```julia
using SpinGlassPEPS

ig = ising_graph(instance)
reference = brute_force(ig; num_states = 8)

potts_h = potts_hamiltonian(
    ig;
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)
```

The exhaustive result is useful as an oracle for small instances when validating a PEPS
search. See the main examples for construction of `PEPSNetwork` and `MpsContractor`.
