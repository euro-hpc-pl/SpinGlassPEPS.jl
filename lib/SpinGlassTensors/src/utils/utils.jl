export rank_reveal, unique_dims

import Base.Prehashed
"""
$(TYPEDSIGNATURES)

Reveal ranks and energies in a specified order.

This function calculates and reveals the ranks and energies of a set of states in either the
'PE' (Projector Energy) or 'EP' (Energy Projector) order.

# Arguments:
- `energy`: The energy values of states.
- `order::Symbol`: The order in which to reveal the ranks and energies. 
It can be either `:PE` for 'Projector Energy)' order (default) or `:EP` for 'Energy Projector' order.

# Returns:
- If `order` is `:PE`, the function returns a tuple `(P, E)` where:
  - `P`: A permutation matrix representing projectors.
  - `E`: An array of energy values.
- If `order` is `:EP`, the function returns a tuple `(E, P)` where:
  - `E`: An array of energy values.
  - `P`: A permutation matrix representing projectors.
"""
function rank_reveal(energy, order = :PE) #TODO: add type
    @assert order ∈ (:PE, :EP)
    dim = order == :PE ? 1 : 2
    E, idx = unique_dims(energy, dim)
    P = identity.(idx)
    order == :PE ? (P, E) : (E, P)
end

@generated function unique_dims(A::AbstractArray{T,N}, dim::Integer) where {T,N}
    quote
        1 <= dim <= $N || return copy(A)
        hashes = zeros(UInt, axes(A, dim))

        # Compute hash for each row
        k = 0
        @nloops $N i A d -> (
            if d == dim
                k = i_d
            end
        ) begin
            @inbounds hashes[k] = hash(hashes[k], hash((@nref $N A i)))
        end

        # Collect index of first row for each hash
        uniquerow = similar(Array{Int}, axes(A, dim))
        firstrow = Dict{Prehashed,Int}()
        for k in axes(A, dim)
            uniquerow[k] = get!(firstrow, Prehashed(hashes[k]), k)
        end
        uniquerows = collect(values(firstrow))

        # Check for collisions
        collided = falses(axes(A, dim))
        @inbounds begin
            @nloops $N i A d -> (
                if d == dim
                    k = i_d
                    j_d = uniquerow[k]
                else
                    j_d = i_d
                end
            ) begin
                if (@nref $N A j) != (@nref $N A i)
                    collided[k] = true
                end
            end
        end

        if any(collided)
            nowcollided = similar(BitArray, axes(A, dim))
            while any(collided)
                # Collect index of first row for each collided hash
                empty!(firstrow)
                for j in axes(A, dim)
                    collided[j] || continue
                    uniquerow[j] = get!(firstrow, Prehashed(hashes[j]), j)
                end
                for v ∈ values(firstrow)
                    push!(uniquerows, v)
                end

                # Check for collisions
                fill!(nowcollided, false)
                @nloops $N i A d -> begin
                    if d == dim
                        k = i_d
                        j_d = uniquerow[k]
                        (!collided[k] || j_d == k) && continue
                    else
                        j_d = i_d
                    end
                end begin
                    if (@nref $N A j) != (@nref $N A i)
                        nowcollided[k] = true
                    end
                end
                (collided, nowcollided) = (nowcollided, collided)
            end
        end

        (@nref $N A d -> d == dim ? sort!(uniquerows) : (axes(A, d))),
        indexin(uniquerow, uniquerows)
    end
end
