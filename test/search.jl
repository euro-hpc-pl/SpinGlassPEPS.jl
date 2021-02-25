using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Low energy spectrum for pathological instance is correct" begin
    # m = 3
    # n = 4
    # t = 3

    m = 2
    n = 2

    β = 1.

    L = 4 #n * m * t
    num_states = 5

    ground_energy = 16.4

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    #instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"
    instance = "$(@__DIR__)/instances/4_001.txt"

    ig = ising_graph(instance, L)
    # update_cells!(
    #    ig,
    #    rule = square_lattice((m, n, t)),
    # )

    fg = factor_graph(
        ig,
        energy=energy,
        spectrum=full_spectrum,
    )

    for origin ∈ (:NW,)# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)


        #∂σ = generate_boundary(peps, σ, i, j)
    
        sol = low_energy_spectrum(peps, num_states)

        # println(sol.probabilities)
        # println(sol.states)
        # println(sol.largest_discarded_probability)
        # @test sol.energies[1] ≈ ground_energy
    end
end
