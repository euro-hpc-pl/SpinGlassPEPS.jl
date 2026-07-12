# ising.jl

**function energy(state_code, graph)**
    
    - state_code: state code for which the energyis to be calculated.
    - graph: graph of the ising model.
    Returns the state energy.


**function energy_qubo(state_code, graph)**
    
    - state_code: state code for which the energy expressed in qubo is to be calculated.
    - graph: graph of the ising model.
    Returns the state energy expressed as QUBO.

**function kernel(graph, energies)**

    - graph: graph of the ising model.
    - energies: array filled with zeros. Each array index represents the state of the system.
    Returns energies for every state.

**function kernel_qubo(graph, energies)**

    - graph: graph of the ising model.
    - energies: array filled with zeros. Each array index represents the state of the system.
    Returns energies expressed as QUBO for every state.

**function kernel_part(graph, energies, part_lst, part_st)**
    
    - graph: graph of the ising model.
    - energies: array filled with zeros. Each array index represents the state of the system.
    - part_lst: list for collecting partial energy results
    - part_st: list for collecting partial state results
    Returns energies for every state.

**function kernel_bucket(graph, energies, idx)**

    - graph: graph of the ising model.
    - energies: array filled with zeros. Each array index represents the state of the system.
    - idx: list for collecting partial energy results
    Returns energies for given indexes.

**function exhaustive_search(ig::IsingGraph)**

    - ig::IsingGraph: graph of ising model represented by IsingGraph structure.
    Returns energies and states for provided model by brute-forece alorithm based on GPU.

**function partial_exhaustive_search(ig::IsingGraph)**

    - ig::IsingGraph: graph of ising model represented by IsingGraph structure.
    Returns partial results for energies and states for provided model by brute-force algorithm based on GPU.

**function exhaustive_search_bucket(ig::IsingGraph, how_many = 8)**

    - ig::IsingGraph: graph of ising model represented by IsingGraph structure.
    - how_many: number of states.
    Returns energies and states for provided model by brute-forece alorithm supported by bucket selection based on GPU.

**function max_chunk_size()**

    Returns the maximum chunk size for the algorithm supported by bucket selection.
