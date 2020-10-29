export MPS_from_gates, unique_neighbors
export MPSControl

struct MPSControl 
    max_bond::Int
    var_ϵ::Number
    max_sweeps::Int
end


function _spectrum(ρ::MPS, k::Int) 
    # ρ needs to be in the right canonical form
    l = length(ρ)

    T = eltype(ρ)
    L = ones(T, 1, 1)

    idx = Dict(1 => -1, 2 => 1)
    left_env = Vector{Matrix{T}}(undef, 2*k)
    
    marginal_pdo = Vector{Number}(undef, 2*k)
    partial_states = Vector{Vector{Int}}(undef, 2*k)

    for i ∈ 1:l
        M = ψ[i]
        
        for (j, (state, L)) ∈ enumerate(partial_config)

            @tensor LL[x, y, σ] := M[γ, σ, x] * L[γ, α] * M[α, σ, y] order = (α, γ) 
            @cast P[σ] := sum(x) LL[x, x, σ]

            marginal_pdo[i] = p[1]
            marginal_pdo[k+i] = p[2]

            for (m, p) ∈ enumerate(P)

                #partial_states[n] = push!()
                marginal_pdo[i] = p
            end    

            perm = sortperm(marginal_pdo) 
            partial_config = partial_config[perm[1:k]]
        end

    end

    pdo
end

function unique_neighbors(ig::MetaGraph, i::Int)
    nbrs = neighbors(ig::MetaGraph, i::Int)
    filter(j -> j > i, nbrs)
end

function _apply_bias!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int)
    M = ψ[i]
    if has_prop(ig, i, :h)
        h = get_prop(ig, i, :h)
        v = [exp(-0.5 * dβ * h * σ) for σ ∈ [-1, 1]]
        @cast M[x, σ, y] = M[x, σ, y] * v[σ]  
    end 
    ψ[i] = M
end

function _apply_exponent!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int, j::Int)
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

function _apply_projector!(ψ::AbstractMPS, i::Int)
    δ = I(2)
    M = ψ[i]

    if i == 1
        @cast M̃[a, σ, (y, b)] := δ[σ, y] * δ[1, y] * M[a, σ, b]
    else   
        @cast M̃[(x, a), σ, (y, b)] := δ[σ, y] * δ[x, y] * M[a, σ, b]
    end 
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, i::Int)    
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

function MPS_from_gates(ig::MetaGraph, mps::MPSControl, gibbs::GibbsControl)
    L = nv(ig)

    # control for MPS
    Dcut = mps.max_bond
    tol = mps.var_ϵ
    max_sweeps = mps.max_sweeps

    # control for Gibbs state
    β = gibbs.β
    schedule = gibbs.β_schedule

    @assert β ≈ sum(schedule) "Incorrect β schedule."

    # prepare ~ Hadamard state as MPS
    prod_state = fill([1., 1.], nv(ig))
    ρ = MPS(prod_state)

    for dβ ∈ schedule, i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i) 

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            for j ∈ nbrs 
                _apply_exponent!(ρ, ig, dβ, i, j) 
            end

            _apply_projector!(ρ, i)

            for l ∈ setdiff(1:L, union(i, nbrs)) 
                _apply_nothing!(ρ, l) 
            end
        end

        # reduce bond dimension
        if bond_dimension(ρ) > Dcut
            ρ = compress(ρ, Dcut, tol, max_sweeps) 
        end
    end
    ρ
end