using MetaGraphs
using Graphs
using CSV

enum(vec) = Dict(v => i for (i, v) ∈ enumerate(vec))

@testset "Lattice graph" begin
    m = 4
    n = 4
    t = 4
    L = 128

    instance = "$(@__DIR__)/instances/chimera_droplets/$(L)power/001.txt"

    for T ∈ [Float16, Float32, Float64]
        ig = ising_graph(T, instance)

        potts_h = potts_hamiltonian(
            ig,
            2,
            cluster_assignment_rule = super_square_lattice((m, n, 2 * t)),
        )

        @test collect(vertices(potts_h)) == [(i, j) for i ∈ 1:m for j ∈ 1:n]

        clv = []
        cle = []
        rank = rank_vec(ig)

        for v ∈ vertices(potts_h)
            cl = get_prop(potts_h, v, :cluster)
            push!(clv, vertices(cl))
            push!(cle, collect(edges(cl)))

            @test rank_vec(cl) == get_prop.(Ref(ig), vertices(cl), :rank)
        end

        # Check if graph is factored correctly
        @test isempty(intersect(clv...))
        @test isempty(intersect(cle...))
    end
end

@testset "Factor graph builds on pathological instance" begin
    m = 3
    n = 4
    t = 3
    L = n * m * t

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

    ising = CSV.File(instance, types = [Int, Int, Float64], header = 0, comment = "#")

    couplings = Dict((i, j) => v for (i, j, v) ∈ ising)

    cedges = Dict(
        ((1, 1), (1, 2)) => [(1, 4), (1, 5), (1, 6)],
        ((1, 1), (2, 1)) => [(1, 13)],
        ((1, 2), (1, 3)) => [(4, 7), (5, 7), (6, 8), (6, 9)],
        ((1, 2), (2, 2)) => [(6, 16), (6, 18), (5, 16)],
        ((2, 1), (2, 2)) => [(13, 16), (13, 18)],
        ((2, 2), (3, 2)) => [(18, 28)],
        ((3, 2), (3, 3)) => [
            (28, 31),
            (28, 32),
            (28, 33),
            (29, 31),
            (29, 32),
            (29, 33),
            (30, 31),
            (30, 32),
            (30, 33),
        ],
    )

    cells = Dict(
        (1, 1) => [1],
        (1, 2) => [4, 5, 6],
        (1, 3) => [7, 8, 9],
        (1, 4) => [],
        (2, 1) => [13],
        (2, 2) => [16, 18],
        (2, 3) => [],
        (2, 4) => [],
        (3, 1) => [],
        (3, 2) => [28, 29, 30],
        (3, 3) => [31, 32, 33],
        (3, 4) => [],
    )

    d = 2
    rank = Dict(c => fill(d, length(idx)) for (c, idx) ∈ cells if !isempty(idx))
    bond_dimensions = [2, 2, 8, 4, 2, 2, 8]

    for T ∈ [Float16, Float32, Float64]
        ig = ising_graph(T, instance)
        @test eltype(ig) == T

        potts_h = potts_hamiltonian(
            ig,
            spectrum = full_spectrum,
            cluster_assignment_rule = super_square_lattice((m, n, t)),
        )

        for (bd, e) in zip(bond_dimensions, edges(potts_h))
            ipl, en, ipr =
                get_prop(potts_h, e, :ipl), get_prop(potts_h, e, :en), get_prop(potts_h, e, :ipr)
            pl = get_projector!(get_prop(potts_h, :pool_of_projectors), ipl, :CPU)
            pr = get_projector!(get_prop(potts_h, :pool_of_projectors), ipr, :CPU)

            @test minimum(size(en)) == bd
            @test maximum(pl) == size(en, 1)
            @test maximum(pr) == size(en, 2)
        end

        for ((i, j), cedge) ∈ cedges
            ipl, en, ipr = get_prop(potts_h, i, j, :ipl),
            get_prop(potts_h, i, j, :en),
            get_prop(potts_h, i, j, :ipr)
            pl = get_projector!(get_prop(potts_h, :pool_of_projectors), ipl, :CPU)
            pr = get_projector!(get_prop(potts_h, :pool_of_projectors), ipr, :CPU)
            base_i = all_states(rank[i])
            base_j = all_states(rank[j])

            idx_i = enum(cells[i])
            idx_j = enum(cells[j])

            # Change it to test if energy is calculated using passed 'energy' function
            energy = zeros(T, prod(rank[i]), prod(rank[j]))

            for (ii, σ) ∈ enumerate(base_i), (jj, η) ∈ enumerate(base_j)
                eij = zero(T)
                for (k, l) ∈ values(cedge)
                    kk = enum(cells[i])[k]
                    ll = enum(cells[j])[l]
                    s = σ[idx_i[k]]
                    r = η[idx_j[l]]
                    J = couplings[k, l]
                    eij += s * J * r
                end
                energy[ii, jj] = eij
            end
            @test eltype(energy) == T == eltype(en)
            @test energy ≈ en[pl, pr]
        end
        @testset "each cluster comprises expected cells" begin
            for v ∈ vertices(potts_h)
                cl = get_prop(potts_h, v, :cluster)
                @test issetequal(vertices(cl), cells[v])
            end
        end
        @testset "each edge comprises expected bunch of edges from source Ising graph" begin
            for e ∈ edges(potts_h)
                outer_edges = get_prop(potts_h, e, :outer_edges)
                @test issetequal(
                    cedges[(src(e), dst(e))],
                    [(src(oe), dst(oe)) for oe ∈ outer_edges],
                )
            end
        end
    end
end

function create_example_potts_hamiltonian(::Type{T}) where {T}
    J12 = -1
    h1 = 1 / 2
    h2 = 0.75

    D = Dict((1, 2) => J12, (1, 1) => h1, (2, 2) => h2)
    ig = ising_graph(T, D)

    potts_hamiltonian(
        ig,
        Dict((1, 1) => 2, (1, 2) => 2),
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => (1, 1), 2 => (1, 2)),
    )
end

potts_h_state_to_spin =
    [([1, 1], [-1, -1]), ([1, 2], [-1, 1]), ([2, 1], [1, -1]), ([2, 2], [1, 1])]

@testset "Decoding solution gives correct spin assignment" begin

    for T ∈ [Float16, Float32, Float64]
        potts_h = create_example_potts_hamiltonian(T)
        @test all(eltype(get_prop(potts_h, e, :en)) == T for e ∈ edges(potts_h))
        for (state, spin_values) ∈ potts_h_state_to_spin
            d = decode_potts_hamiltonian_state(potts_h, state)
            states = collect(values(d))[collect(keys(d))]
            @test states == spin_values
        end
    end
end

"""
Instance below looks like this:

1 -- 2 -- 3
|    |    |
4 -- 5 -- 6
|    |    |
7 -- 8 -- 9

And we group the following spins together: [1, 2, 4, 5], [3, 6], [7, 8], [9].
"""
function create_larger_example_potts_hamiltonian()
    instance = Dict(
        (1, 1) => 0.5,
        (2, 2) => 0.25,
        (3, 3) => 0.3,
        (4, 4) => 0.1,
        (5, 5) => 0.0,
        (6, 6) => -2.0,
        (7, 7) => -1.0,
        (8, 8) => 2.0,
        (9, 9) => 3.1,
        (1, 2) => -1.0,
        (2, 3) => 1.0,
        (4, 5) => 0.5,
        (5, 6) => -0.3,
        (7, 8) => 0.1,
        (8, 9) => 2.2,
        (1, 4) => -1.7,
        (4, 7) => 1.2,
        (2, 5) => 0.2,
        (5, 8) => 0.3,
        (3, 6) => 1.1,
        (6, 9) => 0.7,
    )
    ig = ising_graph(instance)

    assignment_rule = Dict(
        1 => (1, 1),
        2 => (1, 1),
        4 => (1, 1),
        5 => (1, 1),
        3 => (1, 2),
        6 => (1, 2),
        7 => (2, 1),
        8 => (2, 1),
        9 => (2, 2),
    )

    potts_h = potts_hamiltonian(
        ig,
        Dict{NTuple{2,Int},Int}(),
        spectrum = full_spectrum,
        cluster_assignment_rule = assignment_rule,
    )

    ig, potts_h
end

function potts_hamiltonian_energy(potts_h, state)
    # This is highly inefficient, but simple, which makes it suitable for testing.
    # If such a function is needed elsewhere, we need to implement it properly.
    total_en = 0

    # Collect local terms from each cluster
    for (s, v) ∈ zip(state, vertices(potts_h))
        total_en += get_prop(potts_h, v, :spectrum).energies[s]
    end

    # Collect inter-cluster terms
    for edge ∈ edges(potts_h)
        i, j = potts_h.reverse_label_map[src(edge)], potts_h.reverse_label_map[dst(edge)]
        ipl, en, ipr = get_prop(potts_h, edge, :ipl),
        get_prop(potts_h, edge, :en),
        get_prop(potts_h, edge, :ipr)
        pl = get_projector!(get_prop(potts_h, :pool_of_projectors), ipl, :CPU)
        pr = get_projector!(get_prop(potts_h, :pool_of_projectors), ipr, :CPU)
        edge_energy = en[pl, pr]
        total_en += edge_energy[state[i], state[j]]
    end
    total_en
end

@testset "Decoding solution gives spins configuration with corresponding energies" begin
    ig, potts_h = create_larger_example_potts_hamiltonian()

    # Corresponding bases sizes for each cluster are 16, 4, 4, 2.
    all_states = [[i, j, k, l] for i ∈ 1:16 for j ∈ 1:4 for k ∈ 1:4 for l ∈ 1:2]

    for state ∈ all_states
        d = decode_potts_hamiltonian_state(potts_h, state)
        spins = zeros(length(d))
        for (k, v) ∈ d
            spins[k] = v
        end
        σ = [Int.(spins)]
        @test potts_hamiltonian_energy(potts_h, state) ≈ energy(σ, ig)[]
    end
end
