export MPSControl
export solve, solve_new
export MPS2

struct MPSControl
    max_bond::Int
    var_ϵ::Number
    max_sweeps::Int
    β::Number
    dβ::Number
end

_make_left_env(ψ::AbstractMPS, k::Int) = ones(eltype(ψ), 1, 1, k)
_make_left_env_new(ψ::AbstractMPS, k::Int) = ones(eltype(ψ), 1, k)
_make_LL(ψ::AbstractMPS, b::Int, k::Int, d::Int) = zeros(eltype(ψ), b, b, k, d)
_make_LL_new(ψ::AbstractMPS, b::Int, k::Int, d::Int) = zeros(eltype(ψ), b, k, d)

# ρ needs to be ∈ the right canonical form
# function solve(ψ::AbstractMPS, keep::Int)
#     @assert keep > 0 "Number of states has to be > 0"
#     T = eltype(ψ)

#     keep_extra = keep
#     pCut = prob = 0.
#     k = 1

#     if keep < prod(rank(ψ))
#         keep_extra += 1
#     end

#     states = fill([], 1, k)
#     left_env = _make_left_env(ψ, k)

#     for (i, M) ∈ enumerate(ψ)
#         _, d, b = size(M)

#         pdo = zeros(T, k, d)
#         LL = _make_LL(ψ, b, k, d)
#         config = zeros(Int, i, k, d)

#         for j ∈ 1:k
#             L = left_env[:, :, j]

#             for σ ∈ local_basis(d)
#                 m = idx(σ)
#                 LL[:, :, j, m] = M[:, m, :]' * (L * M[:, m, :])
#                 pdo[j, m] = tr(LL[:, :, j, m])
#                 config[:, j, m] = vcat(states[:, j]..., σ)
#             end
#         end

#         perm = collect(1: k * d)
#         k = min(k * d, keep_extra)

#         if k >= keep_extra
#             partialsortperm!(perm, vec(pdo), 1:k, rev=true)
#             prob = vec(pdo)[perm]
#             pCut < last(prob) ? pCut = last(prob) : ()
#         end

#         @cast A[α, β, (l, d)] |= LL[α, β, l, d]
#         left_env = A[:, :, perm]

#         @cast B[α, (l, d)] |= config[α, l, d]
#         states = B[:, perm]
#     end
#     states[:, 1:keep], prob[1:keep], pCut
# end


# ψ needs to be ∈ the right canonical form
function solve(ψ::AbstractMPS, keep::Int)
    @assert keep > 0 "Number of states has to be > 0"
    T = eltype(ψ)

    keep_extra = keep
    lpCut = -1000 # do not like this!
    k = 1

    # this is not elegant
    if keep < prod(rank(ψ))
        keep_extra += 1
    end

    lprob = zeros(T, k)
    states = fill([], 1, k)
    left_env = _make_left_env_new(ψ, k)

    for (i, M) ∈ enumerate(ψ)
        _, d, b = size(M)

        pdo = ones(T, k, d)
        lpdo = zeros(T, k, d)
        LL = _make_LL_new(ψ, b, k, d)
        config = zeros(Int, i, k, d)

        for j ∈ 1:k
            L = left_env[:, j]

            for σ ∈ local_basis(d)
                m = idx(σ)
                LL[:, j, m] = L' * M[:, m, :]
                pdo[j, m] = dot(LL[:, j, m], LL[:, j, m])
                config[:, j, m] = vcat(states[:, j]..., σ)
                LL[:, j, m] = LL[:, j, m] / sqrt(pdo[j, m])
            end
            pdo[j, :] = pdo[j, :] / sum(pdo[j, :])
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
    states', lprob, lpCut
end

function _apply_bias!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int)
    M = ψ[i]
    d = size(M, 2)

    h = get_prop(ig, i, :h)

    v = exp.(-0.5 * dβ * h * local_basis(ψ, i))
    @cast M[x, σ, y] = M[x, σ, y] * v[σ]
    ψ[i] = M
end

function _apply_exponent!(ψ::AbstractMPS, ig::MetaGraph, dβ::Number, i::Int, j::Int, last::Int)
    M = ψ[j]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))

    J = get_prop(ig, i, j, :J)
    C = exp.(-0.5 * dβ * J * local_basis(ψ, i) * local_basis(ψ, j)')

    if j == last
        @cast M̃[(x, a), σ, b] := C[x, σ] * M[a, σ, b]
    else
        @cast M̃[(x, a), σ, (y, b)] := C[x, σ] * D[x, y] * M[a, σ, b]
    end

    ψ[j] = M̃
end

function _apply_projector!(ψ::AbstractMPS, i::Int)
    M = ψ[i]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))

    @cast M̃[a, σ, (y, b)] := D[σ, y] * M[a, σ, b]
    ψ[i] = M̃
end

function _apply_nothing!(ψ::AbstractMPS, l::Int, i::Int)
    M = ψ[l]
    D = typeof(M).name.wrapper(I(physical_dim(ψ, i)))

    @cast M̃[(x, a), σ, (y, b)] := D[x, y] * M[a, σ, b]
    ψ[l] = M̃
end


function multiply_purifications(χ::T, ϕ::T, L::Int) where {T <: AbstractMPS}
    S = promote_type(eltype(χ), eltype(ϕ))
    ψ = T.name.wrapper(S, L)

    for i ∈ 1:L
        A1 = χ[i]
        A2 = ϕ[i]

        @cast B[(l, x), σ, (r, y)] := A1[l, σ, r] * A2[x, σ, y]
        ψ[i] = B
    end
    ψ
end

_holes(l::Int, nbrs::Vector) = setdiff(l+1 : last(nbrs), nbrs)

function _apply_layer_of_gates(ig::MetaGraph, ρ::AbstractMPS, control::MPSControl, dβ::Number)
    L = nv(ig)
    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    for i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i)
        is_right = false
        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            _apply_projector!(ρ, i)

            for j ∈ nbrs
                _apply_exponent!(ρ, ig, dβ, i, j, last(nbrs))
            end

            for l ∈ _holes(i, nbrs)
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

function MPS(ig::MetaGraph, control::MPSControl)

    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    schedule = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps
    rank = get_prop(ig, :rank)

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(rank)
    is_right = true
    @info "Sweeping through β and σ" schedule
    for dβ ∈ schedule
        ρ = _apply_layer_of_gates(ig, ρ, control, dβ)
    end
    ρ
end

function MPS(ig::MetaGraph, control::MPSControl, type::Symbol)
    L = nv(ig)
    Dcut = control.max_bond
    tol = control.var_ϵ
    max_sweeps = control.max_sweeps
    dβ = control.dβ
    β = control.β
    @info "Set control parameters for MPS" Dcut tol max_sweeps
    rank = get_prop(ig, :rank)

    @info "Preparing Hadamard state as MPS"
    ρ = HadamardMPS(rank)
    is_right = true
    @info "Sweeping through β and σ" dβ

    if type == :log
        k = ceil(log2(β/dβ))
        dβmax = β/(2^k)
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
        ρ
    elseif type == :lin
        k = β/dβ
        dβmax = β/k
        ρ = _apply_layer_of_gates(ig, ρ, control, dβmax)
        ρ0 = copy(ρ)
        for j ∈ 1:k
            ρ = multiply_purifications(ρ, ρ0, L)
            if bond_dimension(ρ) > Dcut
                @info "Compresing MPS" bond_dimension(ρ), Dcut
                ρ = compress(ρ, Dcut, tol, max_sweeps)
                is_right = true
            end
        end
    end
    ρ

end
