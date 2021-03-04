using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Simplest possible system of two spins" begin
    #
    # ----------------- Ising model ----------------------
    #
    # E = -1.0 * s1 * s2 + 0.5 * s1 + 0.75 * s2
    #
    # states   -> [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    # energies -> [-2.25, 0.25, 0.75, 1.25]
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
    h1 = 0.5
    h2 = 0.75

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

    # read in pure Ising
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

#   for origin ∈ (:NW, :SW, :WS, :WN) #, :NE, :EN, :SE, :ES)
    for origin ∈ (:NE, :EN, :SE, :ES)

        peps = PepsNetwork(m, n, fg, β, origin, control_params)

        println("$(origin) -> ", peps.i_max, " ", peps.j_max)

        if origin ∈ (:NW, :SW)
            @testset "has properly built PEPS tensors given origin at $(origin)" begin
                p1, e, p2 = get_prop(fg, 1, 2, :split)

                v1 = [exp(-β * h1 * σ) for σ ∈ [-1, 1]]
                v2 = [exp(-β * h2 * σ) for σ ∈ [-1, 1]]

                @cast A1[_, _, r, _, σ] |= v1[σ] * p1[σ, r]
                @cast A2[l, _, _, _, σ] |= v2[σ] * exp.(-β * (e * p2))[l, σ]

                R = PEPSRow(peps, 1)

                @test R[1] ≈ A1
                @test R[2] ≈ A2

                @testset "which produces correct Gibbs state given the origin at $(origin)" begin  
                    @reduce C[σ, η] := sum(l) A1[1, 1, l, 1, σ] * A2[l, 1, 1, 1, η]
                    @test C / sum(C) ≈ reshape(gibbs_tensor(ig, β), 2, 2)
                end
            end
        end

        # solve the problem using B & B
        sol = low_energy_spectrum(peps, num_states)

        @testset "has correct spectrum given the origin at $(origin)" begin  
            @test sol.energies ≈ exact_spectrum.energies

            for (σ, η) ∈ zip(exact_spectrum.states, sol.states)
                @test map(i -> i == -1 ? 1 : 2, σ) == η
            end

            @test sol.largest_discarded_probability === -Inf
        end

        println()
        println("approx: ", sol.energies)
        println("exact:  ", exact_spectrum.energies)
        println("--------------------------------------------")
        println("exact:  ", sol.states) 
        println("approx: ", exact_spectrum.states)
        println()
    end
end
