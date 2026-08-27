using Test

@testset "Published API compatibility" begin
    params = SearchParameters(; max_states = 7, cut_off_prob = 1e-4)
    @test params.max_states == 7
    @test params.cutoff_prob == 1e-4
    @test_throws ArgumentError SearchParameters(; cutoff_prob = 0.1, cut_off_prob = 0.2)

    droplets = SingleLayerDroplets(10.0, 2, :hamming, :RMF)
    @test droplets.max_energy == 10.0
    @test droplets.min_size == 2
    @test droplets.metric === :hamming
    @test droplets.mode === :RMF

    @test SpinGlassEngine._canonical_merge_probability(nothing, :nofit) === :none
    @test SpinGlassEngine._canonical_merge_probability(nothing, :fit) === :median
    @test SpinGlassEngine._canonical_merge_probability(nothing, :python) === :tnac4o
    for strategy in (:none, :median, :tnac4o)
        @test SpinGlassEngine._canonical_merge_probability(nothing, strategy) === strategy
    end
    @test SpinGlassEngine._canonical_merge_probability(:median, nothing) === :median
    @test_throws ArgumentError SpinGlassEngine._canonical_merge_probability(:invalid, nothing)
    @test_throws ArgumentError SpinGlassEngine._canonical_merge_probability(nothing, :invalid)
    @test_throws ArgumentError SpinGlassEngine._canonical_merge_probability(:median, :nofit)

    @test branch_states([1, 2], [Int[]]) == [[1], [2]]
    @test all(state -> state isa Vector{Int}, branch_states([1, 2], [[3, 4]]))
end

@testset "Contractor compatibility and exact probability helpers" begin
    m, n, t = 3, 1, 1
    instance = joinpath(@__DIR__, "instances", "pathological", "chim_1_3_1.txt")
    potts_h = potts_hamiltonian(
        ising_graph(instance);
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
    net = PEPSNetwork{SquareSingleNode{EnergyGauges},Dense,Float64}(
        m,
        n,
        potts_h,
        rotation(0),
    )
    params = MpsParameters{Float64}(; bond_dim = 4, num_sweeps = 1)
    ctr = MpsContractor{SVDTruncate,NoUpdate,Float64}(
        net,
        params;
        onGPU = false,
        beta = 1.0,
        graduate_truncation = false,
    )

    droplets = SingleLayerDroplets(10.0, 2, :hamming, :RMF)
    @test merge_branches(ctr; merge_type = :nofit, update_droplets = droplets) isa Function
    @test merge_branches(ctr; merge_type = :fit) isa Function
    @test merge_branches(ctr; merge_type = :python) isa Function
    @test_throws ArgumentError merge_branches(ctr; merge_prob = :invalid)
    @test_throws ArgumentError merge_branches(ctr; merge_type = :invalid)

    search_order = net.vertex_map.(first(nodes_search_order_Mps(net)))
    decoded = decode_state(net, [1, 2], true)
    @test decoded == Dict(search_order[1] => 1, search_order[2] => 2)

    energies, states = exact_spectrum(potts_h)
    weights = exp.(-ctr.beta .* energies)
    first_node = first(search_order)
    expected = [
        sum(weights[i] for i in eachindex(states) if states[i][first_node] == state)
        for state in 1:cluster_size(potts_h, first_node)
    ]
    expected ./= sum(expected)

    @test exact_conditional_probability(ctr, Int[]) ≈ expected
    @test isapprox(exact_marginal_probability(ctr, [1]), expected[1])
end
