using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Low energy spectrum for pathological instance is correct" begin

    #      Grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3

    D = Dict((1, 2) => -0.9049, 
             (2, 3) =>  0.2838, 

             (3, 3) => -0.7928, 
             (2, 2) =>  0.1208, 
             (1, 1) => -0.3342
    )

    m, n = 1, 2
    L = 4
    β = 1.
    num_states = 8

    ig = ising_graph(D, L)

    update_cells!(
        ig,
        rule = Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2),
    )

    fg = factor_graph(
        ig,
        Dict(1 => 4, 2 => 2),
        energy = energy,
        spectrum = full_spectrum,
    )

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    sp = brute_force(ig; num_states=num_states)
    sp = brute_force(ig; num_states=num_states)

    for origin ∈ (:NW,)# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)
    
        sol = low_energy_spectrum(peps, num_states)

        println(sol.probabilities)
        println(sol.states)
        println(sol.largest_discarded_probability)

        # @test sol.energies[1] ≈ ground_energy
    end
end
