using LinearAlgebra
using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Simplest possible system of two spins" begin
    #
    # ----------------- Ising model ------------------
    #
    # E = -1.0 * s1 * s2 + 0.5 * s1 + 0.75 * s2
    #
    # states   -> [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    # energies -> [-2.25, 0.25, 0.75, 1.25]
    #
    # -------------------------------------------------
    #         Grid
    #     A1    |    A2
    #           |
    #       1 - | - 2
    # -------------------------------------------------

    # Model's parameters
    J12 = -1.0
    h1 = 0.5
    h2 = 0.75

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
    ig = ising_graph(D)

    # construct factor graph with no approx
    fg = factor_graph(
        ig,
        Dict(1 => 2, 2 => 2),
        energy = energy,
        spectrum = full_spectrum,
        cluster_assignment_rule = Dict(1 => 1, 2 => 2), # treat it as a grid with 1 spin cells
    )

    # set parameters to contract exactely
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    # get BF results for comparison
    exact_spectrum = brute_force(ig; num_states=num_states)
    ϱ = gibbs_tensor(ig, β)

    # split on the bond
    p1, e, p2 = get_prop.(Ref(fg), 1, 2, (:pl, :en, :pr))

    @testset "has correct energy on the bond" begin
        en = [ J12 * σ * η for σ ∈ [-1, 1], η ∈ [-1, 1]]
        @test en ≈ p1 * (e * p2)
        @test p1 ≈ p2 ≈ I
    end

    for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PEPSNetwork(m, n, fg, β, origin, control_params)

        @testset "has properly built PEPS tensors given origin at $(origin)" begin

            # horizontal alignment - 1 row, 2 columns
            if peps.i_max == 1 && peps.j_max == 2
                @test origin ∈ (:NW, :SW, :SE, :NE)

                l, k = peps.map[1, 1], peps.map[1, 2]

                v1 = [exp(-β * D[l, l] * σ) for σ ∈ [-1, 1]]
                v2 = [exp(-β * D[k, k] * σ) for σ ∈ [-1, 1]]

                @cast A[_, _, r, _, σ] |= v1[σ] * p1[σ, r]
                en = e * p2 .- minimum(e)
                @cast B[l, _, _, _, σ] |= v2[σ] * exp.(-β * en)[l, σ]

                @reduce ρ[σ, η] := sum(l) A[1, 1, l, 1, σ] * B[l, 1, 1, 1, η]
                if l == 2 ρ = ρ' end

                R = PEPSRow(peps, 1)
                @test [R[1], R[2]] ≈ [A, B]

            # vertical alignment - 1 column, 2 rows
            elseif peps.i_max == 2 && peps.j_max == 1
                @test origin ∈ (:WN, :WS, :ES, :EN)

                l, k = peps.map[1, 1], peps.map[2, 1]

                v1 = [exp(-β * D[l, l] * σ) for σ ∈ [-1, 1]]
                v2 = [exp(-β * D[k, k] * σ) for σ ∈ [-1, 1]]

                @cast A[_, _, _, d, σ] |= v1[σ] * p1[σ, d]
                en = e * p2 .- minimum(e)
                @cast B[_, u, _, _, σ] |= v2[σ] * exp.(-β * en)[u, σ]

                @reduce ρ[σ, η] := sum(u) A[1, 1, 1, u, σ] * B[1, u, 1, 1, η]
                if l == 2 ρ = ρ' end

                @test PEPSRow(peps, 1)[1] ≈ A
                @test PEPSRow(peps, 2)[1] ≈ B
            end

            @testset "which produces correct Gibbs state" begin
                @test ϱ ≈ ρ / sum(ρ)
            end
        end

        # solve the problem using B & B
        sol = low_energy_spectrum(peps, num_states)

        @testset "has correct spectrum given the origin at $(origin)" begin
             for (σ, η) ∈ zip(exact_spectrum.states, sol.states)
                 for i ∈ 1:peps.i_max, j ∈ 1:peps.j_max
                    v = j + peps.j_max * (i - 1)
                     # 1 --> -1 and 2 --> 1
                     @test (η[v] == 1 ? -1 : 1) == σ[v]
                end
             end
             @test sol.energies ≈ exact_spectrum.energies
             @test sol.largest_discarded_probability === -Inf
        end
    end
end
