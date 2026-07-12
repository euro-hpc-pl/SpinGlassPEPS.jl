using SpinGlassNetworks
using LabelledGraphs
using Graphs
using MetaGraphs
using Logging

"""
Instance below looks like this:

1 -- 2 -- 3
|
4 -- 5 -- 6
|
7 -- 8 -- 9
"""
function create_larger_example_potts_hamiltonian_tree()
    instance = Dict(
        (1, 1) => 0.5,
        (2, 2) => 0.25,
        (3, 3) => 0.3,
        (4, 4) => 0.1,
        (5, 5) => -0.1,
        (6, 6) => 0.1,
        (7, 7) => 0.0,
        (8, 8) => 0.1,
        (9, 9) => 0.01,
        (1, 2) => -1.0,
        (2, 3) => 1.0,
        (1, 4) => 1.0,
        (4, 5) => 1.0,
        (5, 6) => 1.0,
        (4, 7) => 1.0,
        (7, 8) => 1.0,
        (8, 9) => 1.0,
    )

    ig = ising_graph(instance)

    assignment_rule = Dict(
        1 => (1, 1, 1),
        2 => (1, 2, 1),
        3 => (1, 3, 1),
        4 => (2, 1, 1),
        5 => (2, 2, 1),
        6 => (2, 3, 1),
        7 => (3, 1, 1),
        8 => (3, 2, 1),
        9 => (3, 3, 1),
    )

    potts_h = potts_hamiltonian(
        ig,
        Dict{NTuple{3,Int},Int}(),
        spectrum = full_spectrum,
        cluster_assignment_rule = assignment_rule,
    )

    ig, potts_h
end

ig, potts_h = create_larger_example_potts_hamiltonian_tree()
beta = 0.1
iter = 0
beliefs = belief_propagation(potts_h, beta; iter = iter)

for v in vertices(potts_h)
    en = get_prop(potts_h, v, :spectrum).energies
    println("vertex ", v, " energy = ", en .- minimum(en), " bp = ", beliefs[v])
end
