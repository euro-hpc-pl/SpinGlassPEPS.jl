using SpinGlassExhaustive

function exp_exhaustive_search()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 

  ig = ising_graph(graph_to_dict(cu_graph))    
    
  exhaustive_search(ig)
end

function exp_bucket_exhaustive_search()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 

  ig = ising_graph(graph_to_dict(cu_graph))    
    
  exhaustive_search_bucket(ig)
end

function exp_partial_exhaustive_search()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 

  ig = ising_graph(graph_to_dict(cu_graph))    
    
  partial_exhaustive_search(ig)
end

function exp_qubo()
  N = 8
  graph = generate_random_graph(N)
  qubo = graph_to_qubo(graph)

  cu_qubo = qubo |> cu 

  k = 2

  energies = CUDA.zeros(2^N)

  threadsPerBlock::Int64 = 2^k
  blocksPerGrid::Int64 = 2^(N-k)

  @cuda blocks=blocks threads=threads kernel_qubo(cu_qubo, energies)

  # sort!(energies)
  # or
  states = sortperm(energies)
  energies[states]

  offset = get_energy_offset(Array(cu_graph))
  Array(sort!(energies)).-offset
end

function exp_ising()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 
  
  k = 2

  energies = CUDA.zeros(2^N)

  threadsPerBlock::Int64 = 2^k
  blocksPerGrid::Int64 = 2^(N-k)

  @cuda blocks=blocks threads=threads kernel(cu_graph, energies)

  sort!(energies)
 
  # or
  # states = sortperm(energies)
  # energies[states]

end

function exp_ising_part()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 
  
  k = 2

  energies = CUDA.zeros(2^N)
  part_st = CUDA.zeros(2^(N-k))
  part_lst = CUDA.zeros(2^(N-k))

  threadsPerBlock::Int64 = 2^k
  blocksPerGrid::Int64 = 2^(N-k)

  @cuda blocks=blocks threads=threads kernel_part(cu_graph, energies, part_lst, part_st)

  sort!(part_lst)
 
  # or
  # idx = sortperm(part_lst)
  # states = part_st[idx]
  # part_lst[idx]

end