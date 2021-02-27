using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "update_energy correctly updates the energy" begin
    #m = 3
    #n = 3
    #t = 1 
    m = 3
    n = 4
    t = 3

    β = 1.

    L = n * m * t
    num_states = 4

    #instance = "$(@__DIR__)/instances/$(L)_001.txt"
    instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

    ig = ising_graph(instance, L)
    update_cells!(
       ig,
       rule = square_lattice((m, n, t)),
    )
    println("vertices ", vertices(ig))
    #println(get_prop(ig, 2, 2, :edge))
    println(Cluster(ig, 1))
    println(Cluster(ig, 2))

    fg = factor_graph(
        ig,
        energy=energy,
        spectrum=full_spectrum,
    )

    println(get_prop(fg, 1, 2, :edge))

    control_params = Dict(
        "bond_dim" => typemax(Int),
        "var_tol" => 1E-8,
        "sweeps" => 4.
    )

    for origin ∈ (:NW,)# :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin, control_params)
        eng = update_energy(peps, ig,  [2, 1])

        println("eng ", eng)
        println("size eng ", size(eng))
    end

end

#=
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

         println(sol.probabilities)
         println(sol.states)
         println(sol.largest_discarded_probability)
        # @test sol.energies[1] ≈ ground_energy
    end
end
=#