export square_lattice

 function square_lattice(size::NTuple{5, Int})  
    m, um, n, un, t = size  
    rule = Dict()

    linear_new = LinearIndices((1:n, 1:m))
    linear_old = LinearIndices((1:t, 1:un, 1:n, 1:um, 1:m))
    
    for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
        #old = (i-1) * um * n * un * t + (ui-1) * n * un * t + (j-1) * un * t + (uj-1) * t + k
        #new = (i-1) * m + j
        old = linear_old[k, uj, j, ui, i]
        new = linear_new[j, i]
        push!(rule, old => new)
    end
    rule
end

function square_lattice(size::NTuple{3, Int})  
    m, n, t = size  
    square_lattice((m, 1, n, 1, t))
end