export MPS_from_gates, unique_neighbors

function unique_neighbors(ig::MetaGraph, i::Int)
    nbrs = neighbors(ig::MetaGraph, i::Int)
    return filter(j -> j > i, nbrs)
end

function _apply_bias!(ψ::MPS, ig::MetaGraph, dβ::T, i::Int) where {T <: Number}
    M = ψ[i]
    if has_prop(ig, i, :h)
        h = get_prop(ig, i, :h)
        v = [exp(-0.5 * dβ * h * σ) for σ ∈ [-1, 1]]
        @cast M[x, σ, y] = M[x, σ, y] * v[σ]  
    end 
    ψ[i] = M
end

function _apply_exponent!(ψ::MPS, ig::MetaGraph, dβ::T, i::Int, j::Int) where {T <: Number}
    δ = I(2)
    M = ψ[j]

    J = get_prop(ig, i, j, :J) 
    C = [exp(-0.5 * dβ * k * J * l) for k ∈ [-1, 1], l ∈ [-1, 1]]

    if j == length(ψ)
        @cast M̃[(x, a), σ, b] := C[x, σ] * δ[x, 1] * M[a, σ, b]  
    else
        @cast M̃[(x, a), σ, (y, b)] := C[x, σ] * δ[x, y] * M[a, σ, b]                     
    end     
    ψ[j] = M̃
end

function _apply_projector!(ψ::MPS, i::Int)
    δ = I(2)
    M = ψ[i]

    if i == 1
        @cast M̃[a, σ, (y, b)] := δ[σ, y] * δ[1, y] * M[a, σ, b]
    else   
        @cast M̃[(x, a), σ, (y, b)] := δ[σ, y] * δ[x, y] * M[a, σ, b]
    end 
    ψ[i] = M̃
end

function _apply_nothing!(ψ::MPS, i::Int)    
    δ = I(2)       
    M = ψ[i] 

    if i == 1
        @cast M̃[a, σ, (y, b)] := δ[1, y] * M[a, σ, b] 
    elseif i == length(ψ)
        @cast M̃[(x, a), σ, b] := δ[x, 1] * M[a, σ, b] 
    else    
        @cast M̃[(x, a), σ, (y, b)] := δ[x, y] * M[a, σ, b] 
    end
    ψ[i] = M̃    
end

function MPS_from_gates(ig::MetaGraph, dβ::T, Dcut::Int=32, var_tol::T=1E-8) where T <: Number 
    L = nv(ig)
    all = 1:L

    # prepare Hadamard state as MPS
    prod_state = [[1, 1] / sqrt(2) for _ ∈ 1:nv(ig)]
    ρ = MPS(prod_state)
   
    # apply gates
    for i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i) 

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            for j ∈ nbrs 
                _apply_exponent!(ρ, ig, dβ, i, j) 
            end

            _apply_projector!(ρ, i)
            rest = setdiff(all, union(i, nbrs))
        else 
            rest = all
        end    

        for l ∈ rest 
            _apply_nothing!(ρ, l) 
        end

        # reduce bond dimension
        #if bond_dimension(ρ) > Dcut compress!(ρ, Dcut, var_tol) end
    end
    return ρ
end