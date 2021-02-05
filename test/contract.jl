@testset "peps_contract correctly collapse the peps network" begin

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
    L = 3
    β = 1.

    ig = ising_graph(D, L)

    update_cells!(
        ig,
        rule = Dict(1 => 1, 2 => 1, 3 => 2),
    )

    fg = factor_graph(
        ig,
        Dict(1 => 4, 2 => 2),
        energy = energy,
        spectrum = full_spectrum,
    )

    for i ∈ 1:4, j ∈ 1:2
        
        cfg = Dict(1 => i, 2 => j)

        Z = []
        for origin ∈ (:NW, :SW, :WS, :WN) 
            peps = PepsNetwork(m, n, fg, β, origin)
            p = peps_contract(peps, cfg)
            push!(Z, p)
        end

        # they all should be the same
        @test all(x -> x ≈ first(Z), Z)

        # the exact Gibbs state
        states = collect.(all_states(rank_vec(ig)))
        ρ = exp.(-β .* energy.(states, Ref(ig)))    
        ϱ = reshape(ρ, (4, 2))

        # probabilities should agree
        if isempty(cfg)
            @test first(Z) ≈ sum(ρ)
        else
            @test first(Z) ≈ ϱ[cfg[1], cfg[2]]
        end 
    end
end        