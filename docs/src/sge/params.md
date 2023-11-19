# Contracting PEPS tensor network
Once we construct the tensor network, we can proceed with its contraction. The first step involves preparing structures to store information about the approximate contraction using the MPS-MPO method and the exploration of states through the branch-and-bound algorithm.

```@raw html
<img src="../images/contract.pdf" width="200%" class="center"/>
```

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
In the domain of matrix operations, our package recognizes the significance of two primary approaches: 
* `Dense` 
* `Sparse`. 
The latter, referred to as sparsity, plays a pivotal role in various computational contexts. Frequently, the matrices involved in our calculations exhibit sparse characteristics, wherein a significant portion of their elements is zero. To accommodate this, our package offers the flexibility to choose the `Sparse` mode. This option not only optimizes memory usage but also substantially enhances computational efficiency, particularly when dealing with matrices characterized by a scarcity of non-zero entries. By selecting the sparse mode, users can leverage the inherent structure of these matrices, streamlining computations, and expediting solutions for intricate problems. This feature underscores our commitment to providing users with the tools to tailor their computational strategies based on the nature of the matrices involved, ensuring optimal performance across diverse scenarios.

# Layout 
Different decompositions of the network into MPS:
* `GaugesEnergy`
* `EnergyGauges`
* `EngGaugesEng`

```@raw html
<img src="../images/layout.pdf" width="200%" class="center"/>
```

# Lattice transformations
The PEPS tensor network within our package stands out for its remarkable versatility, offering users the ability to undergo diverse transformations to meet distinct computational requirements. Notably, users can apply `rotations`, occurring in multiples of $\frac{\pi}{2}$ radians, and `reflections` along various axes. These transformations include rotations and reflections around the horizontal (x), vertical (y), diagonal, and antidiagonal axes. This comprehensive set of transformations empowers researchers to meticulously adjust the orientation and arrangement of the tensor network, providing the means to optimize it for specific problem-solving strategies or align it with the geometrical considerations of their chosen application domain.

```@docs
all_lattice_transformations
LatticeTransformation
rotation
reflection
```

# Gauge 
* NoUpdate
