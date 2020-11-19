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
    
    keep_extra = keep
    pCut = prob = 0.
    k = 1

    if keep < (*)(rank(ψ)...)
        keep_extra += 1
    end
 
    states = fill([], 1, k)
    left_env = ones(T, 1, 1, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, β = size(M)

        pdo = zeros(T, k, d)
        pcond = zeros(T, k, d)
        LL = zeros(T, β, β, k, d)
        LLnew = zeros(T, β, β, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k 
            L = left_env[:, :, j]

            for σ ∈ local_basis(d)
                m = idx(σ)

                LL[:, :, j, m] = M[:, m, :]' * (L * M[:, m, :])
                pdo[j, m] = tr(LL[:, :, j, m])
                pcond[j,m] = pdo[j, m]^2
                config[:, j, m] = vcat(states[:, j]..., σ)
                LLnew[:, :, j, m] = LL[:, :, j, m]/pdo[j,m]
            end
            L = LLnew
        end
        perm = collect(1: k * d)
        k = min(k * d, keep_extra)

        if k >= keep_extra
            partialsortperm!(perm, vec(pcond), 1:k, rev=true)   
            prob = vec(pcond)[perm]
            pCut < last(prob) ? pCut = last(prob) : ()
        end

        @cast A[α, β, (l, d)] |= LLnew[α, β, l, d]
        left_env = A[:, :, perm] 

        @cast B[β, (l, d)] |= config[β, l, d]
        states = B[:, perm]
        println(prob)
    end
    states[:, 1:keep], prob[1:keep], pCut
end

function _apply_bias!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int)
    M = ψ[i]
    d = size(M, 2)

    h = get_prop(ig, i, :h)
    v = [exp(0.5 * dβ * h * σ) for σ ∈ local_basis(d)]

    @cast M[x, σ, y] = M[x, σ, y] * v[σ]  
    ψ[i] = M
end

function _apply_exponent!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int, j::Int)
    M = ψ[j]

    d = size(M, 2)
    basis = local_basis(d)

    J = get_prop(ig, i, j, :J)  
    C = [exp(0.5 * dβ * k * J * l) for k ∈ basis, l ∈ basis]
    D = Array(I(d))

    δ = j == length(ψ) ? D[:, 1:1] : D

    @cast M̃[(x, a), σ, (y, b)] := C[σ, x] * δ[x, y] * M[a, σ, b]                      
    ψ[j] = M̃
end

function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]
    D = Array(I(size(M, 2)))

    δ = i == 1 ? D[1:1,:] : D

    @cast M̃[(x, a), σ, (y, b)] := D[σ, y] * δ[x, y] * M[a, σ, b]
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, i::Int) 
    M = ψ[i] 
    D = Array(I(size(M, 2)))

    if i == 1  δ = D[1:1,:]  elseif  i == length(ψ)  δ = D[:,1:1]  else  δ = D end  

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