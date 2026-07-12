@testset "Compare ising kernel with naive approach" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    ig = ising_graph(graph_to_dict(cu_graph))    
    
    res_naive = SpinGlassNetworks.brute_force(ig)
    res_ising_bucket = exhaustive_search(ig)
    
    @test CUDA.@allowscalar res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

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
    blocks = cld(N, threads)
  
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
    
    CUDA.@allowscalar @test res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

end 
