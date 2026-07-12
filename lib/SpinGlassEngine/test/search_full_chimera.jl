using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

function bench()
    m = 16
    n = 16
    t = 8

    β = 1.0

    L = n * m * t
    num_states = 100

    instance = "$(@__DIR__)/instances/chimera_droplets/2048power/001.txt"

    ig = ising_graph(instance)

    fg = factor_graph(
        ig,
        #spectrum=full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    #for transform ∈ all_lattice_transformations
    peps = PEPSNetwork(m, n, fg, rotation(0), β = β, bond_dim = 32)
    update_gauges!(peps, :rand)
    @time sol = low_energy_spectrum(peps, num_states)#, merge_branches(peps, 1.0))
    println(sol.energies[1:1])
end

bench()
