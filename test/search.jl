using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Low energy spectrum for pathological instance is correct" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = 5

    ground_energy = 16.4

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

    ig = ising_graph(instance, L)


    fg = factor_graph(
        ig,
        energy=energy,
        spectrum=full_spectrum,
        cluster_assignment_rule=chimera_to_square_lattice((m, n, t))
    )

    for origin ∈ (:NW,)# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PEPSNetwork(m, n, fg, β, origin, control_params)
        sol = low_energy_spectrum(peps, num_states)
        println(sol.probabilities)
        println(sol.states)
        println(sol.largest_discarded_probability)
        #@test sol.energies[1] ≈ ground_energy
    end
end
