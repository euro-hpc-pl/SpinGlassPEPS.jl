# naive.jl


**function naive_energy_kernel(J, energies, σ)**

    - J: partial description of graph of the ising model.
    - energies: array filled with zeros. Each array index represents the state of the system.
    - σ: partial description of graph of the ising model.
    Returns the state energy.

** function brute_force(ig::IsingGraph)**

    - ig::IsingGraph: graph of ising model represented by IsingGraph structure.
    Returns energies and states for provided model by naive brute-forece alorithm based on GPU.