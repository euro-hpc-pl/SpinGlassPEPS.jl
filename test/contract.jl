@testset "contract correctly collapse the peps network" begin

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

    m, n, t = 1, 2, 2
    L = 3
    β = 2.

    ig = ising_graph(D, L)

    update_cells!(
        ig,
        rule = Dict(1 => 1, 2 => 1, 3 => 2),
    )

    fg = factor_graph(
        ig,
        Dict(1 => 4, 2 => 2),
        energy = energy,
        spectrum = brute_force,
    )

    config = Dict(1 => 2, 2 => 1)

    for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
        peps = PepsNetwork(m, n, fg, β, origin)
        p = peps_contract(peps, config)
    end
end        