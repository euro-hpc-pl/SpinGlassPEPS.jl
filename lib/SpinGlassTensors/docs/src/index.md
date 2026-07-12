# SpinGlassTensor

Part of [SpinGlassPEPS](https://github.com/euro-hpc-pl/SpinGlassPEPS.jl) package. It constitutes the basis for the preparation of tensors and operations on them.

!!! info
    We don't expect the user to interact with this package, as it is more of a "back-end" type. Nevertheless, we provide API references should the need arise.

This section of the package encompasses supplementary functionalities that serve as support for the main solver. `SpinGlassTensors.jl`:
* includes the creation and manipulation of various types of tensors.
* offers efficient tools for tensor contractions, in particular it handles tensor contractions in a **sparse** and **dense** mode, enabling efficient memory and computational use depending on the problem's structure.
* supports QMps (Matrix Product States) and QMpo (Matrix Product Operators) for representing states and operators in a tensor network framework.
* implements advanced tensor optimization algorithms, including **zipper** algorithm for approximating boundary MPS and **variational** schemes.
* supports computation on both **CPU** and **GPU**, enabling high-performance tensor operations with the `onGPU` flag in core tensor structures.