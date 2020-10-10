dg(M::AbstractArray{T, 4}) where {T} = permutedims(conj.(M), (2, 1, 3, 4))
dg(M::AbstractArray{T, 3}) where {T} = permutedims(conj.(M), (2, 1, 3))