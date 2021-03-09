export chimera_to_square_lattice

 function chimera_to_square_lattice(size::NTuple{5, Int})
    m, um, n, un, t = size
    new = LinearIndices((1:n, 1:m))
    old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))

    Dict(
            old[k, uj, j, ui, i] => new[j, i]
            for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
    )
end

function chimera_to_square_lattice(size::NTuple{3, Int})
    m, n, t = size
    chimera_to_square_lattice((m, 1, n, 1, t))
end
