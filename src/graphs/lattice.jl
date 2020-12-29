export square_lattice

 function square_lattice(size::NTuple{5, Int})  
    m, um, n, un, t = size  
    rule = Dict()

    linear_new = LinearIndices((1:m, 1:n))
    linear_old = LinearIndices((1:m, 1:um, 1:n, 1:un, 1:t))
    
    for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
        old = linear_old[i, ui, j, uj, k]
        new = linear_new[i, j]
        push!(rule, old => new)
    end
    rule
end

function square_lattice(size::NTuple{3, Int})  
    m, n, t = size  
    square_lattice((m, 1, n, 1, t))
end