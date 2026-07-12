# Contracting PEPS tensor network
Once we construct the tensor network, we can proceed with its contraction. The first step involves preparing structures to store information about the approximate contraction using the MPS-MPO method and the exploration of states through the branch-and-bound algorithm.

```@docs
MpsContractor
```

# Structures to store parameters used in branch and bound search
```@docs
MpsParameters
SearchParameters
```

# Strategy 
In the boundary MPS-MPO approach we apply Matrix Product Operator (MPO) to appropriate sites of Matrix Product State (MPS). In this context, the absorption of a MPO into a MPS leads to an exponential growth of the bond dimension. Hence, a truncation scheme is necessary to mitigate this issue and to keep the bond dimension in a reasonable size. 
Our package offers users the flexibility to choose between three distinct methods for optimizing the boundary MPS used in contracting the tensor network: 
* `Zipper`
* `SVDTruncate`.
`Zipper` method combines a zipper scheme of [Ref.](https://arxiv.org/abs/2310.08533). with the standard variational optimization of the resulting MPS [(see Ref.)](https://arxiv.org/abs/0907.2796)
```@raw html
<img src="../images/zipper_final.png" width="200%" class="center"/>
```
With the `SVDTruncate` method, the Matrix Product State (MPS) is systematically constructed row by row, contracted with the Matrix Product Operator (MPO) from the preceding row. The resulting MPS undergoes a Singular Value Decomposition (SVD) to truncate its bond dimension, followed by variational compression. 
```@raw html
<img src="../images/svd_truncate.png" width="50%" class="center"/>
```

# Sparsity  
The `Sparsity` parameter controls whether 
* `Dense` 
or 
* `Sparse` 
tensor representations are used during calculations. `Sparse` tensors are particularly useful for handling large clusters containing around 10 to 20 spins. When bond dimensions increase, constructing PEPS tensors explicitly (triggered by `Sparsity=Dense`) becomes computationally expensive and quickly infeasible. In contrast, setting `Sparsity=Sparse` avoids the direct construction of full tensors. Instead, it performs optimal contractions on smaller tensor structures, which are then combined to contract the entire network efficiently. This approach leverages the internal structure of the individual tensors to reduce computational overhead and memory usage.

# Geometry
One can specify the type of the node used within the tensor networks: 
* `SquareSingleNode`
```@raw html
<img src="../images/square_single.png" width="50%" class="center"/>
```
```@docs
SquareSingleNode
```

* `SquareDoubleNode`
```@raw html
<img src="../images/square_double.png" width="50%" class="center"/>
```
```@docs
SquareDoubleNode
```

* `KingSingleNode`
```@raw html
<img src="../images/square_cross_single.png" width="50%" class="center"/>
```
```@docs
KingSingleNode
```

* `SquareCrossDoubleNode`
```@raw html
<img src="../images/square_cross_double.png" width="50%" class="center"/>
```
```@docs
SquareCrossDoubleNode
```

# Layout 
`SpinGlassPEPS.jl` allows for different decompositions of the PEPS network into MPOs:
* `GaugesEnergy`
* `EnergyGauges`
* `EngGaugesEng`
For complex problems, the solution may depend on the choice of decomposition.

```@raw html
<img src="../images/layout.png" width="200%" class="center"/>
```

# Lattice transformations
Our package provides users with the ability to apply various transformations to the PEPS network, allowing for flexibility in tensor network manipulation. The available transformations include `rotations` by multiples of $\frac{\pi}{2}$ radians and `reflections` along different axes. Specifically, users can apply rotations and reflections around the horizontal (x), vertical (y), diagonal, and antidiagonal axes.

These transformations are useful when contracting PEPS or performing searches from different lattice sites. For instance, the `transform` parameter allows the user to rotate the quasi-2D graph, which influences the order of sweeping through local variables during branch-and-bound search. By rotating the tensor network, the search and contraction process can start from different positions on the 2D grid, improving the stability and robustness of the results.

In practice, searches can be performed across all eight possible transformations (four rotations and four reflections, `all_lattice_transformations`) of the 2D grid, comparing the energies obtained for each configuration to identify the most optimal outcome.

```@raw html
<img src="../images/trans.png" width="200%" class="center"/>
```
```@docs
all_lattice_transformations
rotation
reflection
```

# Gauge 
Currently only `NoUpdate` mode is supported.