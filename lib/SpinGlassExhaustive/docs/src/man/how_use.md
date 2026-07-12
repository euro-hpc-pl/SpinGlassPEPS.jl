```@setup SpinGlassExhaustive
using SpinGlassExhaustive

import Pkg;
Pkg.add("CUDA")
using CUDA

Pkg.add("SpinGlassEngine")
using SpinGlassEngine
```

# How use SpinGlassExhaustive
Within the package, you can use one of the following functions to exhaustive search of the solution space of the Ising model

- `brute_force` - it provides solution based on CPU brute force searching;
- `exhaustive_search` - it provides solution based on GPU brute force searching;
- `exhaustive_search_bucket` - it provides solution based on GPU brute force searching supported by bucket-sort algorithm;
- `partial_exhaustive_search` - it provides partial solution based on GPU brute force searching;

Each of these functions is called with ising_graph object as input.

Below is an example of using the above functions for a random graph generated with the function `generate_random_graph`.


```@repl SpinGlassExhaustive
N = 8

graph = SpinGlassExhaustive.generate_random_graph(N)

cu_graph = graph |> cu;

ig = SpinGlassEngine.ising_graph(SpinGlassExhaustive.graph_to_dict(graph))

SpinGlassEngine.brute_force(ig)

SpinGlassExhaustive.exhaustive_search(ig)

SpinGlassExhaustive.exhaustive_search_bucket(ig)

SpinGlassExhaustive.partial_exhaustive_search(ig)
```

