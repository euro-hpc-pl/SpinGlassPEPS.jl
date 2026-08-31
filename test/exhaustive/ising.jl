@testset "GPU exhaustive search returns the complete 10-spin spectrum" begin
    N = 10
    graph = zeros(Float32, N, N)
    for i in 1:N
        graph[i, i] = Float32(mod(3i, 7) - 3) / 4
        for j in i+1:N
            graph[i, j] = Float32(mod(5i + 3j, 11) - 5) / 8
        end
    end

    ig = ising_graph(graph_to_dict(graph))
    cpu = SpinGlassNetworks.brute_force(ig; num_states = 2^N)
    gpu = exhaustive_search(ig)

    gpu_states = Array(gpu.states)
    gpu_energies = Array(gpu.energies)
    cpu_energy_by_state = Dict(zip(cpu.states_int, cpu.energies))

    @test sort(gpu_states) == collect(0:2^N-1)
    @test all(
        isapprox(energy, cpu_energy_by_state[state]; rtol = 1f-6, atol = 1f-6) for
        (energy, state) in zip(gpu_energies, gpu_states)
    )
end

@testset "Exhaustive searches preserve zero-based state codes" begin
    ig = ising_graph(
        Float32,
        Dict((1, 1) => 0.25f0, (2, 2) => -0.5f0, (1, 2) => 1.25f0),
    )
    expected = Dict(
        code => only(
            SpinGlassNetworks.energy(
                [2 .* digits(code; base = 2, pad = 2) .- 1],
                ig,
            ),
        ) for code in 0:3
    )

    # The default requests eight results, but a two-spin problem has only four.
    for result in (exhaustive_search(ig), exhaustive_search_bucket(ig))
        states = Array(result.states)
        energies = Array(result.energies)

        @test sort(states) == collect(0:3)
        @test all(
            isapprox(en, expected[state]; rtol = 1f-6) for
            (en, state) in zip(energies, states)
        )
    end


    @test_throws ArgumentError exhaustive_search_bucket(ig, 0)
end

# @testset "Compare ising kernel returning partial result with naive approach" begin
#     N = 8
#     graph = generate_random_graph(N)
#     cu_graph = graph |> cu 
    
#     ig = ising_graph(graph_to_dict(cu_graph))    
    
#     res_naive = SpinGlassNetworks.brute_force(ig)
#     res_ising_bucket = partial_exhaustive_search(ig)
    
#     @test res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

# end 

@testset "Check conversion of qubo solution to ising model" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    qubo = graph_to_qubo(graph)

    cu_qubo = qubo |> cu 

  
    energies = CUDA.zeros(2^N)
    qubo_energies = CUDA.zeros(2^N)
  
    threads = 512
    blocks = cld(2^N, threads)

    @cuda blocks=blocks threads=threads kernel(cu_graph, energies)
  
    CUDA.@allowscalar cuda_min_energy = sort!(energies)[1]

    @cuda blocks=blocks threads=threads kernel_qubo(cu_qubo, qubo_energies) 

    offset = SpinGlassExhaustive.get_energy_offset(Array(cu_graph))

    @test CUDA.@allowscalar cuda_min_energy[1] ≈ sort!(qubo_energies)[1]-offset

end 

@testset "Compare ising kernel with bucket sort returning result with naive approach" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    ig = ising_graph(graph_to_dict(cu_graph))    
    
    res_naive = SpinGlassNetworks.brute_force(ig)
    res_ising_bucket = exhaustive_search_bucket(ig)

    # GPU kernels accumulate in Float32; compare at Float32 precision.
    CUDA.@allowscalar @test isapprox(res_ising_bucket.energies[1], minimum(res_naive.energies); rtol = 1e-6)

end 
