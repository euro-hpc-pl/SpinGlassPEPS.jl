export zephyr_to_linear, unique_neighbors, load_openGM

import Base.Prehashed
using HDF5


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

# @generated function unique_dims(A::AbstractArray{T,N}, dim::Integer) where {T,N}
#     quote
#         1 <= dim <= $N || return copy(A)
#         hashes = zeros(UInt, axes(A, dim))

#         # Compute hash for each row
#         k = 0
#         @nloops $N i A d->(if d == dim; k = i_d; end) begin
#             @inbounds hashes[k] = hash(hashes[k], hash((@nref $N A i)))
#         end

#         # Collect index of first row for each hash
#         uniquerow = similar(Array{Int}, axes(A, dim))
#         firstrow = Dict{Prehashed,Int}()
#         for k = axes(A, dim)
#             uniquerow[k] = get!(firstrow, Prehashed(hashes[k]), k)
#         end
#         uniquerows = collect(values(firstrow))

#         # Check for collisions
#         collided = falses(axes(A, dim))
#         @inbounds begin
#             @nloops $N i A d->(if d == dim
#                 k = i_d
#                 j_d = uniquerow[k]
#             else
#                 j_d = i_d
#             end) begin
#                 if (@nref $N A j) != (@nref $N A i)
#                     collided[k] = true
#                 end
#             end
#         end

#         if any(collided)
#             nowcollided = similar(BitArray, axes(A, dim))
#             while any(collided)
#                 # Collect index of first row for each collided hash
#                 empty!(firstrow)
#                 for j = axes(A, dim)
#                     collided[j] || continue
#                     uniquerow[j] = get!(firstrow, Prehashed(hashes[j]), j)
#                 end
#                 for v ∈ values(firstrow)
#                     push!(uniquerows, v)
#                 end

#                 # Check for collisions
#                 fill!(nowcollided, false)
#                 @nloops $N i A d->begin
#                     if d == dim
#                         k = i_d
#                         j_d = uniquerow[k]
#                         (!collided[k] || j_d == k) && continue
#                     else
#                         j_d = i_d
#                     end
#                 end begin
#                     if (@nref $N A j) != (@nref $N A i)
#                         nowcollided[k] = true
#                     end
#                 end
#                 (collided, nowcollided) = (nowcollided, collided)
#             end
#         end

#         (@nref $N A d->d == dim ? sort!(uniquerows) : (axes(A, d))), indexin(uniquerow, uniquerows)
#     end
# end

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
function load_openGM(
    fname::String,
    Nx::Union{Integer,Nothing} = nothing,
    Ny::Union{Integer,Nothing} = nothing,
)
    file = h5open(fname, "r")

    file_keys = collect(keys(read(file)))
    data = read(file[file_keys[1]])
    H = collect(Int64, data["header"])
    F = Array{Int64}(data["factors"])
    J = Array{Int64}(data["function-id-16000"]["indices"])
    V = Array{Real}(data["function-id-16000"]["values"])
    N = Array{Int64}(data["numbers-of-states"])

    if isnothing(Nx) || isnothing(Ny)
        filename, _ = splitext(basename(fname))
        Nx, Ny = benchmark_names[filename]
    end

    F = reverse(F)
    factors = Dict()

    while length(F) > 0
        f1 = pop!(F)
        z1 = pop!(F)
        nn = pop!(F)
        n = []

        for _ = 1:nn
            tt = pop!(F)
            ny, nx = divrem(tt, Nx)
            push!(n, ny, nx)
        end
        if length(n) == 4
            if abs(n[1] - n[3]) + abs(n[2] - n[4]) ∉ [1, 2] || (
                abs(n[1] - n[3]) + abs(n[2] - n[4]) == 2 &&
                (abs(n[1] - n[3]) == 2 || abs(n[2] - n[4] == 2))
            )
                throw(ErrorException("Not nearest neighbour or diagonal neighbors"))
            end
        end

        if length(n) == 2
            if (n[1] >= Ny) || (n[2] >= Nx)
                throw(ErrorException("Wrong size"))
            end
        end

        factors[tuple(n...)] = f1

        if z1 != 0
            throw(ErrorException("Something wrong with the expected convention."))
        end
    end

    J = reverse(J)
    functions = Dict()
    ii = -1
    lower = 0

    while length(J) > 0
        ii += 1
        nn = pop!(J)
        n = []

        for _ = 1:nn
            push!(n, pop!(J))
        end

        upper = lower + prod(n)
        functions[ii] = reshape(V[lower+1:upper], reverse(n)...)'

        lower = upper
    end

    result = Dict(
        "fun" => functions,
        "fac" => factors,
        "N" => reshape(N, (Ny, Nx)),
        "Nx" => Nx,
        "Ny" => Ny,
    )
    result
end

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
