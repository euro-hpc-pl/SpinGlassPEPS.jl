using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Pathological instance" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = 22

    exact_energies = [
        -16.4, -16.4, -16.4, -16.4, -16.1, -16.1, -16.1, -16.1, -15.9,
        -15.9, -15.9, -15.9, -15.9, -15.9, -15.6, -15.6, -15.6, -15.6,
        -15.6, -15.6, -15.4, -15.4
    ]
    exact_states = [ 
        [0, 2, 2, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 2],
        [0, 2, 2, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2],
        [0, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 2],
        [0, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2],
    ]

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

    ig = ising_graph(instance, L)
    update_cells!(
        ig, 
        rule = square_lattice((m, n, t))
    )

    fg = factor_graph(
        ig,
        energy=energy,
        spectrum=full_spectrum,
    )

   for origin ∈ (:NW, )# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)

        # solve the problem using B & B
        sol = low_energy_spectrum(peps, num_states)

        println(sol.energies)
        println(sol.states)
        println(sol.largest_discarded_probability)
    end

end
