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
* `MPSAnnealing`
* `SVDTruncate`.

With the `SVDTruncate` method, the Matrix Product State (MPS) is systematically constructed row by row, contracted with the Matrix Product Operator (MPO) from the preceding row. The resulting MPS undergoes a Singular Value Decomposition (SVD) to truncate its bond dimension, followed by variational compression. 
On the other hand, the `MPSAnnealing` method tailors the construction of MPS based on temperature considerations, with a subsequent variational compression step. 
`Zipper` method combines randomized truncated Singular Value Decomposition (SVD) and a variational
scheme.

# Sparsity 
Our software package acknowledges the importance of two fundamental methodologies in tensor processing
* `Dense` 
* `Sparse`. 
The latter, referred to as sparsity, plays a pivotal role in manipulation on large tensors. To accommodate this, our package offers the flexibility to choose the `Sparse` mode. In this mode, tensors are not explicitly constructed but are stored in structures and represented as blocks, in which not every dimension is contracted. This choice not only optimizes memory utilization but also significantly improves computational efficiency. In the `Dense` mode tensors are build explicitly.

# Geometry
* `SquareSingleNode`
```@docs
SquareSingleNode
```

* `SquareDoubleNode`
```@docs
SquareDoubleNode
```

* `SquareCrossSingleNode`
```@docs
SquareCrossSingleNode
```

* `SquareCrossDoubleNode`
```@docs
SquareCrossDoubleNode
```

# Layout 
`SpinGlassPEPS.jl` allows for different decompositions of the network into MPOs:
* `GaugesEnergy`
* `EnergyGauges`
* `EngGaugesEng`
For complex problems, the solution may depend on the choice of decomposition.

```@raw html
<img src="../images/layout.pdf" width="200%" class="center"/>
```

# Lattice transformations
Our package offers users the ability to undergo diverse transformations of PEPS network. Notably, users can apply `rotations`, occurring in multiples of $\frac{\pi}{2}$ radians, and `reflections` along various axes. These transformations include rotations and reflections around the horizontal (x), vertical (y), diagonal, and antidiagonal axes. Transformations are used to contract PEPS and perform search starting from different sites of the lattice. 
```@raw html
<img src="../images/transform.pdf" width="200%" class="center"/>
```
```@docs
all_lattice_transformations
rotation
reflection
```

# Gauge 
Currently only `NoUpdate` mode is supported.