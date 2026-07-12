```@setup SpinGlassExhaustive
using SpinGlassExhaustive

import Pkg;
Pkg.add("CUDA")
using CUDA

Pkg.add("SpinGlassEngine")
using SpinGlassEngine

Pkg.add("SpinGlassNetworks")
using SpinGlassNetworks
```

# How to use GPU kernels for exaustive search
The section contains examples illustrating how to use CUDA kernels for exhaustive search of the solution space of the Ising model.

**kernel_qubo - returns energies expressed as QUBO for every state**
```@repl SpinGlassExhaustive
N = 8
graph = SpinGlassExhaustive.generate_random_graph(N)
qubo = SpinGlassExhaustive.graph_to_qubo(graph)

cu_qubo = qubo |> cu;

k = 2

energies = CUDA.zeros(2^N);

threads = 512
blocks = cld(N, threads)

@cuda blocks=blocks threads=threads SpinGlassExhaustive.kernel_qubo(cu_qubo, energies);

states = sortperm(energies);
energies[states];

offset = SpinGlassExhaustive.get_energy_offset(Array(graph));
Array(sort!(energies)).-offset
```

**kernel - returns energies for every state**
```@repl SpinGlassExhaustive
N = 8
graph = SpinGlassExhaustive.generate_random_graph(N)
cu_graph = graph |> cu;

k = 2

energies = CUDA.zeros(2^N);

threads = 512
blocks = cld(N, threads)

@cuda blocks=blocks threads=threads SpinGlassExhaustive.kernel(cu_graph, energies);

states = sortperm(energies);
energies[states]
```

**kernel_part - returns the lowest energies for $2^{(N-k)}$ states**
```@repl SpinGlassExhaustive
N = 8
graph = SpinGlassExhaustive.generate_random_graph(N)
cu_graph = graph |> cu;

k = 2

energies = CUDA.zeros(2^N);
part_st = CUDA.zeros(2^(N-k));
part_lst = CUDA.zeros(2^(N-k));

threads = 512
blocks = cld(N, threads)

@cuda blocks=blocks threads=threads SpinGlassExhaustive.kernel_part(cu_graph, energies, part_lst, part_st);

idx = sortperm(part_lst);
states = part_st[idx];
part_lst[idx]
```

**kernel_bucket - returns energies for given indexes**
```@repl SpinGlassExhaustive
N = 8
how_many = 4

graph = SpinGlassExhaustive.generate_random_graph(N)

ig = SpinGlassEngine.ising_graph(SpinGlassExhaustive.graph_to_dict(graph))

L = SpinGlassNetworks.nv(ig)    

σ = CUDA.fill(Int32(-1), L, 2^L);
J = couplings(ig) + SpinGlassNetworks.Diagonal(biases(ig))
J_dev = CUDA.CuArray(J)

N = size(J_dev,1)

chunk_size = SpinGlassExhaustive.max_chunk_size()

if chunk_size > N
    chunk_size = N
end


energies_d = CUDA.zeros(Float64, 2^chunk_size);
lowest_d = CUDA.zeros(Float64, how_many*2);
lowest_states_d = CUDA.zeros(Int64, how_many*2);

k = 2
threads = 512
blocks = cld(chunk_size, threads)



for i in 1:2^(N-chunk_size)

    idx = (i-1)*(2^chunk_size) + 1 
    
    @cuda blocks=blocks threads=threads SpinGlassExhaustive.kernel_bucket(J_dev, energies_d, idx)

    states_d = sortperm(energies_d)[1:how_many]

    if i == 1
        lowest_d[1:how_many] = energies_d[states_d]
        lowest_states_d[1:how_many] = (states_d.+idx)
    else
        lowest_d[how_many+1:2*how_many] = energies_d[states_d]
        lowest_states_d[how_many+1:2*how_many] = (states_d.+idx)

        states = sortperm(lowest_d)
        lowest_d = lowest_d[states]
        lowest_states_d = lowest_states_d[states]
    end 
end

lowest_d[1:how_many], lowest_states_d[1:how_many]
```