@testset "search correctly  " begin
    
    m = 3
    n = 4
    t = 3

    β = 1

    L = m * n * t
    T = Float64

    cut = 2

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

    x, y = m, n

    for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)

        peps = PepsNetwork(x, y, fg, β, origin)
        @test typeof(peps) == PepsNetwork
        sol = low_energy_spectrum(peps, cut)
        
    end
 
end