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

# ρ needs to be in the right canonical form
function spectrum(ψ::MPS, keep::Int) 
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    k = 1
    pCut = prob = 0.

    states = fill([], 1, k)
    left_env = ones(T, 1, 1, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, β = size(M)

        basis = union(-1, 1:d-1)

        pdo = zeros(T, k, d)
        LL = zeros(T, β, β, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k 
            L = left_env[:, :, j]

            for σ ∈ basis
                m = idx(σ)

                LL[:, :, j, m] = M[:, m, :]' * (L * M[:, m, :])
                pdo[j, m] = tr(LL[:, :, j, m])

                config[:, j, m] = vcat(states[:, j]..., σ)
            end  
        end
        k = min(k * d, keep)

        perm = partialsortperm(vec(pdo), 1:k, rev=true)  

        @cast A[α, β, (l, d)] |= LL[α, β, l, d] 
        left_env = A[:, :, perm]

        @cast B[α, (l, d)] |= config[α, l, d]
        states = B[:, perm]

        prob = vec(pdo)[perm]
        #pCut < last(prob) ? pCut = last(prob) : ()
    end
    states, prob# ,pCut
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


    δ = j == length(ψ) ? P' : D
    j == length(ψ) ? δ = P' : δ = D

    @cast M̃[(x, a), σ, (y, b)] := C[σ, x] * δ[x, y] * M[a, σ, b]                      
    ψ[j] = M̃
end

function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]

    δ = i == 1 ? P : D
 
    @cast M̃[(x, a), σ, (y, b)] := D[σ, y] * δ[x, y] * M[a, σ, b]
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, i::Int) 
    M = ψ[i] 
  
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