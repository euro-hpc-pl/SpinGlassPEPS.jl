export MPS_from_gates, unique_neighbors
export MPSControl
export spectrum

export _apply_bias!
export _apply_exponent!
export _apply_projector!
export _apply_nothing!

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
                @info "Probability of spin being up, down" i j k marginal_pdo[j] marginal_pdo[k+j]
                @info "Left environment" left_env[j] left_env[j+k]
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
    h = get_prop(ig, i, :h)
    v = [exp(0.5 * dβ * h * σ) for σ ∈ [-1, 1]]
    @cast M[x, σ, y] = M[x, σ, y] * v[σ]  
    ψ[i] = M
end

function _apply_exponent!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int, j::Int)
    M = ψ[j]

    J = get_prop(ig, i, j, :J) 
    C = [exp(0.5 * dβ * k * J * l) for k ∈ [-1, 1], l ∈ [-1, 1]]

    D = I(2)
    P = [[1.] [0.]]
    if j == length(ψ) δ = P' else δ = D end

    @cast M̃[(x, a), σ, (y, b)] := C[σ, x] * δ[x, y] * M[a, σ, b]                      
    ψ[j] = M̃
end

function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]

    D = I(2)
    P = [[1.] [0.]]
    if i == 1 δ = P else δ = D end
 
    @cast M̃[(x, a), σ, (y, b)] := D[σ, y] * δ[x, y] * M[a, σ, b]
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, i::Int) 
    M = ψ[i] 

    D = I(2)
    P = [[1.] [0.]]     
    if i == 1  δ = P  elseif  i == length(ψ)  δ = P'  else  δ = D end  

    @cast M̃[(x, a), σ, (y, b)] := δ[x, y] * M[a, σ, b] 
    ψ[i] = M̃    
end

function MPS(ig::MetaGraph, mps::MPSControl, gibbs::GibbsControl)
    L = nv(ig)

    Dcut = mps.max_bond
    tol = mps.var_ϵ
    max_sweeps = mps.max_sweeps
    @info "Set control parameters for MPS" Dcut tol max_sweeps

    β = gibbs.β
    schedule = gibbs.β_schedule

    @assert β ≈ sum(schedule) "Incorrect β schedule."

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(L)
    is_right = true

    @info "Sweeping through β and σ" schedule
    for dβ ∈ schedule, i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i) 
        is_right = false

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)

            @info "Applying outgoing gates from $i"
            _apply_projector!(ρ, i)

            for j ∈ nbrs 
                _apply_exponent!(ρ, ig, dβ, i, j) 
            end

            for l ∈ setdiff(1:L, union(i, nbrs))
                _apply_nothing!(ρ, l) 
            end
        end

        if bond_dimension(ρ) > Dcut
            @info "Compresing MPS" bond_dimension(ρ), Dcut
            ρ = compress(ρ, Dcut, tol, max_sweeps) 
            is_right = true
        end
    end

    if !is_right
        canonise!(ρ, :right)
        is_right = true
    end
    ρ
end