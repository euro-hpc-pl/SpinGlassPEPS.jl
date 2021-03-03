using MetaGraphs
using LightGraphs
using GraphPlot
using CSV


@testset "Simplest possible system of two spins" begin
    #
    # ----------------- Ising model ----------------------
    #
    # E = -s1 * s2 + 2 * s1 + s2
    #
    # states   -> [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    # energies -> [-4.0, 0.0, 2.0, 2.0]
    #
    # -----------------------------------------------------
    #         Grid
    #
    #     A1    |    A2
    #           |
    #       1 - | - 2
    # -----------------------------------------------------

    # Model's parameters
    J12 = -1.0
    h1 = 2.0
    h2 = 1.0

    total_energy(σ::Int, η::Int) = J12 * (σ * η) + h1 * σ + h2 * η
    bond_energy(σ::Int, η::Int) = J12 * (σ * η)

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

    # read in pure ising
    ig = ising_graph(D, L)

    # treat it as a grid with 1 spin cells
    update_cells!(
        ig,
        rule = Dict(1 => 1, 2 => 2),
    )

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict(1 => 2, 2 => 2),
        energy = energy,
        spectrum = full_spectrum,
    )

    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    # get BF results for comparison
    exact_spectrum = brute_force(ig; num_states=num_states)

    @testset "has correct energy on the bond" begin
        p1, e, p2 = get_prop(fg, 1, 2, :split)
        en = [ bond_energy(σ, η) for σ ∈ [-1, 1], η ∈ [-1, 1]]
        @test en ≈ p1 * (e * p2)
    end

    for origin ∈ (:NW, )# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)

        @testset "has properly built PEPS tensors" begin

            p1, e, p2 = get_prop(fg, 1, 2, :split)

            v1 = [ exp(-β * h1 * σ) for σ ∈ [-1, 1]]
            v2 = [ exp(-β * h2 * σ) for σ ∈ [-1, 1]]
            println("Projectors: ")
            display(p1)
            println()
            display(p2)
            println()
            @cast A1[_, _, r, _, σ] |= v1[σ] * p1[σ, r]
            @cast A2[l, _, _, _, σ] |= v2[σ] * exp.(-β * (e * p2))[l, σ]

            R = PEPSRow(peps, 1)

            @test R[1] ≈ A1
            @test R[2] ≈ A2

            @cast C[σ, η] := sum(l) A1[1, 1, l, 1, σ] * A2[l, 1, 1, 1, η]
            println("C:")
            display(C)
            println()
            println(C)
            println(size(C))
            println(typeof(C))
            println("size of A1 and A2")
            println(size(A1), size(A2))

        end

        sol = low_energy_spectrum(peps, num_states)


        println(sol.probabilities)
        println(sol.states)
        println(sol.largest_discarded_probability)

        # @test sol.energies[1] ≈ ground_energy
    end

    println(exact_spectrum.states)
    println(exact_spectrum.energies)
end
