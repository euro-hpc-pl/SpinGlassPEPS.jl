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
Our package offers users the flexibility to choose between three distinct methods for contracting the tensor network: 
* `Zipper`
* `MPSAnnealing`
* `SVDTruncate`.

With the `SVDTruncate` method, the Matrix Product State (MPS) is systematically constructed row by row, contracting with the Matrix Product Operator (MPO) from the preceding row. The resulting MPS undergoes a Singular Value Decomposition (SVD) to trim its bond dimension, followed by variational compression. 
On the other hand, the `MPSAnnealing` method tailors the construction of MPS based on temperature considerations, with a subsequent variational compression step. 

These approaches provide users with distinct strategies to efficiently contract the tensor network, catering to different preferences and requirements in the exploration of spin systems within the SpinGlassPEPS package.

# Sparsity 
Our software package acknowledges the importance of two fundamental methodologies in tensor processing
* `Dense` 
* `Sparse`. 
The latter, referred to as sparsity, plays a pivotal role in manipulation on large tensors. To accommodate this, our package offers the flexibility to choose the `Sparse` mode. In this mode, tensors are not explicitly constructed but are storerd in structures and represented as blocks where not every dimension is contracted. This choice not only optimizes memory utilization but also significantly improves computational efficiency. In the `Dense` mode tensors are build explicitly.

# Geometry
* SquareSingleNode
* SquareDoubleNode
* SquareCrossSingleNode
* SquareCrossDoubleNode

# Layout 
Different decompositions of the network into MPS:
* `GaugesEnergy`
* `EnergyGauges`
* `EngGaugesEng`

```@raw html
<img src="../images/layout.pdf" width="200%" class="center"/>
```

# Lattice transformations
Our package offers users the ability to undergo diverse transformations of PEPS network to meet distinct computational requirements. Notably, users can apply `rotations`, occurring in multiples of $\frac{\pi}{2}$ radians, and `reflections` along various axes. These transformations include rotations and reflections around the horizontal (x), vertical (y), diagonal, and antidiagonal axes. 

```@docs
all_lattice_transformations
LatticeTransformation
rotation
reflection
```

# Gauge 
* NoUpdate
