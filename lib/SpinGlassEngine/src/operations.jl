# operations.jl: This file profides functions and structs for perfoming operations
#                on the whole PEPS tensor network ("lattice").

export LatticeTransformation,
    rotation, reflection, all_lattice_transformations, vertex_map, check_bounds

"""
$(TYPEDSIGNATURES)

A struct representing a lattice transformation.

# Fields
- `permutation::NTuple{4, Int}`: A tuple defining a permutation of the vertex labels.
- `flips_dimensions::Bool`: A boolean indicating whether dimension flips are applied.

The `LatticeTransformation` struct defines a transformation that can be applied to the vertices of a lattice. 
It specifies a permutation of vertex labels, allowing for rotations and reflections, as well as an option to flip dimensions. 
"""
struct LatticeTransformation
    permutation::NTuple{4,Int}
    flips_dimensions::Bool
end

# """
# $(TYPEDSIGNATURES)
# """
function Base.:(∘)(op1::LatticeTransformation, op2::LatticeTransformation)
    LatticeTransformation(
        op1.permutation[collect(op2.permutation)],
        op1.flips_dimensions ⊻ op2.flips_dimensions,
    )
end

"""
$(TYPEDSIGNATURES)
Create a reflection transformation along the specified axis.

# Arguments
- `axis::Symbol`: The axis of reflection, which can be one of the following:
  - `:x` for reflection along the x-axis.
  - `:y` for reflection along the y-axis.
  - `:diag` for reflection along the main diagonal.
  - `:antydiag` for reflection along the anti-diagonal.

# Returns
A `LatticeTransformation` object representing the specified reflection transformation.
"""
function reflection(axis::Symbol)
    if axis == :x
        perm = (4, 3, 2, 1)
        flips = false
    elseif axis == :y
        perm = (2, 1, 4, 3)
        flips = false
    elseif axis == :diag
        perm = (1, 4, 3, 2)
        flips = true
    elseif axis == :antydiag
        perm = (3, 2, 1, 4)
        flips = true
    else
        throw(ArgumentError("Unknown reflection axis: $(axis)"))
    end
    LatticeTransformation(perm, flips)
end

"""
$(TYPEDSIGNATURES)
Create a rotation transformation by a specified angle.

# Arguments
- `θ::Int`: The angle of rotation, expressed in degrees. Only integral multiples of 90° can be passed as θ.

# Returns
A `LatticeTransformation` object representing the specified rotation transformation.
"""
function rotation(θ::Int)
    if θ % 90 != 0
        ArgumentError("Only integral multiplicities of 90° can be passed as θ.")
    end
    θ = θ % 360
    if θ == 0
        LatticeTransformation((1, 2, 3, 4), false)
    elseif θ == 90
        LatticeTransformation((4, 1, 2, 3), true)
    else
        rotation(θ - 90) ∘ rotation(90)
    end
end

"""
$(TYPEDSIGNATURES)
Create a bounds-checking function for a lattice of size (m, n).

# Arguments
- `m::Int`: The number of rows in the lattice.
- `n::Int`: The number of columns in the lattice.

# Returns
A bounds-checking function that can be used to ensure that lattice points are within the specified bounds.
"""
function check_bounds(m, n)
    function _check(i, j)
        if i < 1 || i > m || j < 1 || j > n
            throw(ArgumentError("Point ($(i), $(j)) not in $(m)x$(n) lattice."))
        end
        true
    end
end

"""
$(TYPEDSIGNATURES)
Create a vertex mapping function for a lattice transformation.

# Arguments
- `vert_permutation::NTuple{4, Int}`: A permutation of vertex labels, defining a specific lattice transformation.
- `nrows::Int`: The number of rows in the lattice.
- `ncols::Int`: The number of columns in the lattice.
    
# Returns
A vertex mapping function `vmap` that takes a tuple of vertex coordinates and returns their new coordinates after applying the specified lattice transformation.    
"""
function vertex_map(vert_permutation::NTuple{4,Int}, nrows, ncols)
    if vert_permutation == (1, 2, 3, 4) #
        f = (i, j) -> (i, j)
    elseif vert_permutation == (4, 1, 2, 3) # 90 deg rotation
        f = (i, j) -> (nrows - j + 1, i)
    elseif vert_permutation == (3, 4, 1, 2) # 180 deg rotation
        f = (i, j) -> (nrows - i + 1, ncols - j + 1)
    elseif vert_permutation == (2, 3, 4, 1) # 270 deg rotation
        f = (i, j) -> (j, ncols - i + 1)
    elseif vert_permutation == (2, 1, 4, 3) # :y reflection
        f = (i, j) -> (i, ncols - j + 1)
    elseif vert_permutation == (4, 3, 2, 1) # :x reflection
        f = (i, j) -> (nrows - i + 1, j)
    elseif vert_permutation == (1, 4, 3, 2) # :diag reflection
        f = (i, j) -> (j, i)
    elseif vert_permutation == (3, 2, 1, 4) # :antydiag reflection
        f = (i, j) -> (nrows - j + 1, ncols - i + 1)
    else
        ArgumentError("$(vert_permutation) does not define square isometry.")
    end
    vmap(node::NTuple{2,Int}) = f(node[1], node[2])
    vmap(node::NTuple{3,Int}) = (f(node[1], node[2])..., node[3])
    vmap
end

"""
$(TYPEDSIGNATURES)
Create a vertex map function based on a given lattice transformation.

This function generates a vertex map function that can be used to transform lattice vertex coordinates according to a specified lattice transformation. The `trans` argument should be a `LatticeTransformation` object, and `m` and `n` specify the dimensions of the lattice.
    
# Arguments
- `trans::LatticeTransformation`: The lattice transformation to apply.
- `m::Int`: The number of rows in the lattice.
- `n::Int`: The number of columns in the lattice.
    
# Returns
A vertex map function that takes vertex coordinates and returns the transformed coordinates.    
"""
function vertex_map(trans::LatticeTransformation, m::Int, n::Int)
    vertex_map(trans.permutation, m, n)
end

"""
$(TYPEDSIGNATURES)
A tuple containing all possible lattice transformations.

This constant includes rotations at angles 0, 90, 180, and 270 degrees, as well as reflections across the x-axis, y-axis, diagonal, and antidiagonal axes. 
These lattice transformations can be applied to the PEPS lattice to achieve various orientations and reflections in order to start the search on a different sites of lattice.
"""
const all_lattice_transformations =
    (rotation.([0, 90, 180, 270])..., reflection.([:x, :y, :diag, :antydiag])...)
