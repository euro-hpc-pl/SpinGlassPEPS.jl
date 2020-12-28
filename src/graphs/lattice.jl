export Lattice

struct Lattice <: Model
    size::NTuple{5, Int}
    graph::MetaGraph
    
    function Lattice(size::NTuple{5, Int}, ig::MetaGraph)  
        lt = new(size, ig)
        m, um, n, un, t = lt.size  
        
        linear_grid = LinearIndices((1:m, 1:n))
        linear_graph = LinearIndices((1:m, 1:um, 1:n, 1:un, 1:t))
        
        for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
            ijk = linear_graph[i, du, j, uj, k]
            set_prop!(lt, ig[ijk], :cell, linear_grid[i, j])
        end
        lt
    end
end

function cluster!(ig::MetaGraph, size::NTuple{5, Int})  
    m, um, n, un, t = size  
    
    linear_grid = LinearIndices((1:m, 1:n))
    linear_graph = LinearIndices((1:m, 1:um, 1:n, 1:un, 1:t))
        
    for i=1:m, ui=1:um, j=1:n, uj=1:un, k=1:t
        ijk = linear_graph[i, du, j, uj, k]
        set_prop!(ig, ig[ijk], :cell, linear_grid[i, j])
    end
end
