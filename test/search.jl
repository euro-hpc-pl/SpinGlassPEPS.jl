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
    num_states = L^2

    ground_energy = 16.4 
    
    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"
    
    ig = ising_graph(instance, L)
    update_cells!(
       ig,
       rule = square_lattice((m, n, t)),
    )
    
    fg = factor_graph(
        ig,
        energy=energy,
        spectrum=full_spectrum,
    )

    for origin ∈ (:NW,)# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)
        sol = low_energy_spectrum(peps, num_states)
        
        @test sol.energies[1] ≈ ground_energy
    end
end
    
