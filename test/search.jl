
@testset "Low energy spectrum fo pathological instance" begin
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = L^2

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    file = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"
    instance = CSV.File(file, types=[Int, Int, Float64], header=0, comment = "#")
    
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
        sol = low_energy_spectru(peps, num_states)
        
        
    end
    