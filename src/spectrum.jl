export unique_neighbors
export full_spectrum, brute_force
export MPSControl
export solve, solve_new

struct Spectrum
    energies::Array{<:Number}
    states::Array{Vector{<:Number}}
end

struct MPSControl
    max_bond::Int
    var_ϵ::Number
    max_sweeps::Int
    β::Vector
end

# ρ needs to be in the right canonical form
function solve(ψ::MPS, keep::Int)
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    keep_extra = keep
    pCut = prob = 0.
    k = 1

    if keep < prod(rank(ψ))
        keep_extra += 1
    end

    states = fill([], 1, k)
    left_env = ones(T, 1, 1, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, b = size(M)

        pdo = zeros(T, k, d)
        LL = zeros(T, b, b, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k
            L = left_env[:, :, j]

            for σ ∈ local_basis(d)
                m = idx(σ)

                LL[:, :, j, m] = M[:, m, :]' * (L * M[:, m, :])
                pdo[j, m] = tr(LL[:, :, j, m])
                config[:, j, m] = vcat(states[:, j]..., σ)
            end
        end

        perm = collect(1: k * d)
        k = min(k * d, keep_extra)

        if k >= keep_extra
            partialsortperm!(perm, vec(pdo), 1:k, rev=true)
            prob = vec(pdo)[perm]
            pCut < last(prob) ? pCut = last(prob) : ()
        end

        @cast A[α, β, (l, d)] |= LL[α, β, l, d]
        left_env = A[:, :, perm]

        @cast B[α, (l, d)] |= config[α, l, d]
        states = B[:, perm]
    end
    states[:, 1:keep], prob[1:keep], pCut
end

# ρ needs to be in the right canonical form
function solve_new(ψ::MPS, keep::Int) 
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    keep_extra = keep
    lpCut = -1000
    k = 1

    if keep < prod(rank(ψ))
        keep_extra += 1
    end
    lprob = zeros(T, k)
    states = fill([], 1, k)
    left_env = ones(T, 1, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, b = size(M)

        pdo = ones(T, k, d)
        lpdo = zeros(T, k, d)
        LL = zeros(T, b, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k
            L = left_env[:, j]

            for σ ∈ local_basis(d)
                m = idx(σ)
                LL[:, j, m] = L' * M[:, m, :]
                pdo[j, m] = dot(LL[:, j, m], LL[:, j, m])
                config[:, j, m] = vcat(states[:, j]..., σ)
                LL[:, j, m] = LL[:, j, m]/sqrt(pdo[j, m])
            end
            pdo[j, :] = pdo[j, :]/sum(pdo[j, :])
            lpdo[j, :] = log.(pdo[j, :]) .+ lprob[j]
        end

        perm = collect(1 : k * d)
        k = k * d

        if k > keep_extra
            k = keep_extra
            partialsortperm!(perm, vec(lpdo), 1:k, rev=true)
            lprob = vec(lpdo)[perm]
            lpCut < last(lprob) ? lpCut = last(lprob) : ()
        end

        lprob = vec(lpdo)[perm]
        @cast A[α, (l, d)] |= LL[α, l, d]
        left_env = A[:, perm]
        @cast B[β, (l, d)] |= config[β, l, d]
        states = B[:, perm]
    end
    states = states'
    states, lprob, lpCut
end

function _apply_bias!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int)
    M = ψ[i]
    d = size(M, 2)

    h = get_prop(ig, i, :h)
    v = [exp(-0.5 * dβ * h * σ) for σ ∈ local_basis(d)]

    @cast M[x, σ, y] = M[x, σ, y] * v[σ]
    ψ[i] = M
end

function _apply_exponent!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int, j::Int, last::Int)
    M = ψ[j]
    D = I(ψ, i)

    J = get_prop(ig, i, j, :J)
    C = [ exp(-0.5 * dβ * k * J * l) for k ∈ local_basis(ψ, i), l ∈ local_basis(ψ, j) ]

    if j == last
        @cast M̃[(x, a), σ, b] := C[x, σ] * M[a, σ, b]
    else
        @cast M̃[(x, a), σ, (y, b)] := C[x, σ] * D[x, y] * M[a, σ, b]
    end

    ψ[j] = M̃
end

function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]
    D = I(ψ, i)

    @cast M̃[a, σ, (y, b)] := D[σ, y] * M[a, σ, b]
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, l::Int, i::Int)
    M = ψ[l]
    D = I(ψ, i)

    @cast M̃[(x, a), σ, (y, b)] := D[x, y] * M[a, σ, b]
    ψ[l] = M̃
end

_holes(nbrs::Vector) = setdiff(first(nbrs) : last(nbrs), nbrs)


function MPS(ig::MetaGraph, control::MPSControl)
    L = nv(ig)

    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    schedule = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps

    β = get_prop(ig, :β)
    rank = get_prop(ig, :rank)

    @assert β ≈ sum(schedule) "Incorrect β schedule."

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(rank)
    is_right = true

    @info "Sweeping through β and σ" schedule
    for dβ ∈ schedule, i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i)
        is_right = false

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            _apply_projector!(ρ, i)

            for j ∈ nbrs
                _apply_exponent!(ρ, ig, dβ, i, j, last(nbrs))
            end

            for l ∈ _holes(nbrs)
                _apply_nothing!(ρ, l, i) 
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


"""
$(TYPEDSIGNATURES)

Return the low energy spectrum

# Details

Calculates \$k\$ lowest energy states
together with the coresponding energies
of a classical Ising Hamiltonian
"""
function brute_force(ig::MetaGraph; num_states::Int=1)
    cl = Cluster(ig, 0, enum(vertices(ig)), edges(ig))
    brute_force(cl, num_states=num_states)
end


function brute_force(cl::Cluster; num_states::Int=1)
    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    perm = partialsortperm(vec(energies), 1:num_states) 
    Spectrum(energies[perm], σ[perm])
end

function full_spectrum(cl::Cluster)
    σ = collect.(all_states(cl.rank))
    energies = energy.(σ, Ref(cl))
    Spectrum(energies, σ)   
end
