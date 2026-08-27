"""
Instance below looks like this:

1 -- 2
|
3 -- 4

"""
function create_larger_example_potts_hamiltonian_tree_basic()
    instance = Dict(
        (1, 1) => -0.50,
        (2, 2) => 0.25,
        (3, 3) => -0.30,
        (4, 4) => 0.10,
        (1, 2) => -0.23,
        (1, 3) => 1.10,
        (3, 4) => 0.71,
    )

    ig = ising_graph(instance)

    assignment_rule = Dict(1 => (1, 1, 1), 2 => (1, 2, 1), 3 => (2, 1, 1), 4 => (2, 2, 2))

    potts_h = potts_hamiltonian(
        ig,
        Dict{NTuple{3,Int},Int}(),
        spectrum = full_spectrum,
        cluster_assignment_rule = assignment_rule,
    )

    ig, potts_h
end

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
        (1, 1) => 0.53,
        (2, 2) => -0.25,
        (3, 3) => 0.30,
        (4, 4) => -0.10,
        (5, 5) => -0.10,
        (6, 6) => 0.10,
        (8, 8) => 0.10,
        (9, 9) => 0.01,
        (1, 2) => -1.00,
        (2, 3) => 1.00,
        (1, 4) => 0.33,
        (4, 5) => 0.76,
        (5, 6) => -0.45,
        (4, 7) => -0.28,
        (7, 8) => 0.36,
        (8, 9) => -1.07,
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

"""
Instance below looks like this:

123 -- 45 -- 6
|      |
7      8

"""
function create_larger_example_potts_hamiltonian_tree_pathological()
    instance = Dict(
        (1, 1) => 0.52,
        (2, 2) => 0.25,
        (3, 3) => -0.31,
        (4, 4) => 0.17,
        (5, 5) => -0.12,
        (6, 6) => 0.13,
        (7, 7) => 0.00,
        (8, 8) => 0.43,
        (1, 2) => -1.01,
        (1, 3) => 1.00,
        (3, 4) => 0.97,
        (3, 5) => -0.98,
        (5, 6) => 1.00,
        (2, 7) => 0.53,
        (3, 7) => 1.06,
        (5, 8) => -0.64,
    )

    ig = ising_graph(instance)

    assignment_rule = Dict(
        1 => (1, 1),
        2 => (1, 1),
        3 => (1, 1),
        4 => (1, 2),
        5 => (1, 2),
        6 => (1, 3),
        7 => (2, 1),
        8 => (2, 2),
    )

    potts_h = potts_hamiltonian(
        ig,
        Dict{NTuple{2,Int},Int}(),
        spectrum = full_spectrum,
        cluster_assignment_rule = assignment_rule,
    )

    ig, potts_h
end

@testset "Belief propagation" begin
    for (ig, potts_h) ∈ [
        create_larger_example_potts_hamiltonian_tree_basic(),
        create_larger_example_potts_hamiltonian_tree(),
        create_larger_example_potts_hamiltonian_tree_pathological(),
    ]
        for beta ∈ [0.5, 1]
            iter = 16
            beliefs = belief_propagation(potts_h, beta; iter = iter)
            exact_marginal = Dict()
            for k in keys(beliefs)
                push!(
                    exact_marginal,
                    k => [
                        exact_cond_prob(potts_h, beta, Dict(k => a)) for
                        a = 1:length(beliefs[k])
                    ],
                )
            end
            for v in keys(beliefs)
                temp = -log.(exact_marginal[v]) ./ beta
                @test beliefs[v] ≈ temp .- minimum(temp)
            end
        end
    end
end
