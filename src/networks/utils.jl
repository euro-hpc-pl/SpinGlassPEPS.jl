export zephyr_to_linear, unique_neighbors, load_openGM



"""
$(TYPEDSIGNATURES)

Rewriten from Dwave-networkx
m - Grid parameter for the Zephyr lattice.
t - Tile parameter for the Zephyr lattice; must be even.
"""
function zephyr_to_linear(m::Int, t::Int, q::NTuple{5,Int})
    M = 2 * m + 1
    u, w, k, j, z = q
    (((u * M + w) * t + k) * 2 + j) * m + z + 1
end

unique_neighbors(ig::LabelledGraph, i::Int) = filter(j -> j > i, neighbors(ig, i))

"""
$(TYPEDSIGNATURES)
Loads some factored graphs written in openGM format. Assumes rectangular lattice.

Args:
    file_name (str): a path to file with factor graph in openGM format.
    ints Nx, Ny: it is assumed that graph if forming an :math:N_x \times N_y lattice with
        nearest-neighbour interactions only.

Returns:
   dictionary with factors and funcitons defining the energy functional.
"""
function load_openGM end

benchmark_names = Dict(
    "penguin-small" => (240, 320),
    "palm-small" => (240, 360),
    "clownfish-small" => (240, 360),
    "crops-small" => (240, 360),
    "pfau-small" => (240, 320),
    "lake-small" => (240, 360),
    "snail" => (240, 320),
    "fourcolors" => (240, 320),
    "strawberry-glass-2-small" => (320, 240),
)
