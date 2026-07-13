@testset "PEPS networks own independent projector workspaces" begin
    m, n, t = 3, 1, 1
    instance = joinpath(@__DIR__, "instances", "pathological", "chim_$(n)_$(m)_$(t).txt")
    potts_h = potts_hamiltonian(
        ising_graph(instance);
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )

    net1 = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
        m,
        n,
        potts_h,
        rotation(0),
    )
    net2 = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
        m,
        n,
        potts_h,
        rotation(90),
    )

    @test net1.lp !== net2.lp
    @test net1.lp !== projector_pool(potts_h)
    @test net2.lp !== projector_pool(potts_h)
    @test net1.lp.data[:CPU] !== net2.lp.data[:CPU]
    @test all(
        net1.lp.data[:CPU][index] == net2.lp.data[:CPU][index] for
        index in keys(net1.lp.data[:CPU])
    )

    net1.lp.matrices[(:sentinel, :CPU)] = :cached
    @test isempty(net2.lp.matrices)
    @test isempty(projector_pool(potts_h).matrices)
end
