export MPS_from_gates, unique_neighbors
export MPSControl
export spectrum

struct MPSControl 
    max_bond::Int
    var_ϵ::Number
    max_sweeps::Int
end

function spectrum(ρ::MPS, k::Int) 
    # ρ needs to be in the right canonical form
    l = length(ρ)
    T = eltype(ρ)

    @assert k > 0 "Number of states has to be > 0."
    @assert is_right_normalized(ρ)

    left_env = fill(ones(T, 1, 1), 2*k)  
    marginal_pdo = fill(0., 2*k)
    partial_states = fill(Int[], 2*k) 

    pcut_max = 0.

    for i ∈ 1:l
        M = ρ[i]

        for (j, (state, L)) ∈ enumerate(zip(partial_states[1:k], left_env[1:k]))
            for σ ∈ [-1, 1]
                m = idx(σ)
                n = j + (m - 1) * k
                left_env[n] = M[:, m, :]' * (L * M[:, m, :])
                marginal_pdo[n] = tr(left_env[n])
                partial_states[n] = push!(state, σ)
            end  
            
            @debug begin 
                @info "Probability of spin being up, down" i marginal_pdo[j] marginal_pdo[k+j]
                @assert marginal_pdo[j] + marginal_pdo[k+j] ≈ 1
            end
        end

        perm = sortperm(marginal_pdo, rev=true) 
        marginal_pdo = marginal_pdo[perm]

        pcut_max < marginal_pdo[k+1] ? pCutMax = marginal_pdo[k+1] : ()

        partial_states = partial_states[perm]
        left_env = left_env[perm]
    end

    partial_states[1:k], marginal_pdo[1:k], pcut_max
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

function MPS(ig::MetaGraph, mps::MPSControl, gibbs::GibbsControl)
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