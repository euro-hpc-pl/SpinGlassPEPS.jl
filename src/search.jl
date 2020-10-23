

function MPS(ig::MetaGraph) 

    L = nv(ig)
    β = get_prop(ig, :β)
    
    C = I(2)
    δ = I(2)

    # Hadamard state to MPS
    prod_state = [ [1, 1] / sqrt(2) for _ ∈ 1:L]
    ρ = MPS(prod_state)
    
    for dβ ∈ schedule
        for i ∈ vertices(ig)
            for j ∈ neighbors(ig, i)

                J = get_prop(ig, i, j, :J)
                B = [ exp(-dβ * x * J * y) for x ∈ [-1, 1], y ∈ [-1, 1] ]

                M = ρ[i]
                @reduce N[(x, a), (y, b), σ] := B[x, σ] * δ[x, y] * M[a, η, b] 
            end
        end
    end  

    return ρ
end
