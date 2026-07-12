module SpinGlassTensors
using cuTENSOR
using CUDA, CUDA.CUSPARSE
using NNlib
using LinearAlgebra, MKL
using TensorOperations
using LowRankApprox, TSVD
using Memoization
using SparseArrays
using DocStringExtensions
using Base.Cartesian

import Base.Prehashed

CUDA.allowscalar(false)

ArrayorCuArray(A::AbstractArray, onGPU) = onGPU ? CuArray(A) : A

include("projectors.jl")
include("base.jl")
include("linear_algebra_ext.jl")
include("utils/utils.jl")
include("./mps/base.jl")
include("./mps/transpose.jl")
include("./mps/dot.jl")
include("./mps/identity.jl")
include("./mps/utils.jl")
include("./mps/rand.jl")
include("transfer.jl")
include("environment.jl")
include("utils/memory.jl")
include("./mps/canonise.jl")
include("variational.jl")
include("zipper.jl")
include("gauges.jl")
include("contractions/sparse.jl")
include("contractions/dense.jl")
include("contractions/central.jl")
include("contractions/diagonal.jl")
include("contractions/site.jl")
include("contractions/virtual.jl")


end # module
