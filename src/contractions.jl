export left_env, right_env, dot!

# --------------------------- Conventions -----------------------
#
#      MPS          MPS*         MPO       left env     right env
#       2            2            2           - 1          2 -
#   1 - A - 3    1 - B - 3    1 - W - 3      L               R
#                                 4           - 2          1 -
# ---------------------------------------------------------------
#

function LinearAlgebra.dot(ψ::AbstractMPS, state::Union{Vector, NTuple})
    C = I

    for (M, σ) ∈ zip(ψ, state)
        i = idx(σ)
        C = M[:, i, :]' * (C * M[:, i, :])
    end
    tr(C)
end

function LinearAlgebra.dot(ϕ::AbstractMPS, ψ::AbstractMPS)
    T = promote_type(eltype(ψ), eltype(ϕ))
    C = ones(T, 1, 1)

    for i ∈ eachindex(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end

function left_env(ϕ::AbstractMPS, ψ::AbstractMPS)
    l = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))
    S = typeof(similar(ψ[1], T, (1, 1)))
    L = Vector{S}(undef, l+1)
    L[1] = similar(ψ[1], T, (1, 1))
    L[1][1, 1] = one(T)

    for i ∈ 1:l
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        C = L[i]
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    L
end

@memoize function left_env(ϕ::AbstractMPS, σ::Vector{Int})
    l = length(σ)
    if l == 0
        L = [1.]
    else
        m = σ[l]
        L̃ = left_env(ϕ, σ[1:l-1])
        M = ϕ[l]
        @reduce L[x] := sum(α) L̃[α] * M[α, $m, x]
    end
    L
end

# NOT tested yet
function right_env(ϕ::AbstractMPS, ψ::AbstractMPS)
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))
    S = typeof(similar(ψ[1], T, (1, 1)))
    R = Vector{S}(undef, L+1)
    R[end] = similar(ψ[1], T, (1, 1))
    R[end][1, 1] = one(T)

    for i ∈ L:-1:1
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        D = R[i+1]
        @tensor D[x, y] := M[x, σ, α] * D[α, β] * M̃[y, σ, β] order = (β, α, σ)
        R[i] = D
    end
    R
end

@memoize function right_env(ϕ::AbstractMPS{T}, W::AbstractMPO{T}, σ::Union{Vector, NTuple}) where {T}
    l = length(σ)
    #k = length(ϕ)
    k = length(W)
    if l == 0
        R = similar(ϕ[1], T, (1, 1))
        R[1, 1] = one(T)
    else
        m = σ[1]
        R̃ = right_env(ϕ, W, σ[2:l])
        M = ϕ[k-l+1]
        M̃ = W[k-l+1]
        @reduce R[x, y] := sum(α, β, γ) M̃[y, $m, β, γ] * M[x, γ, α] * R̃[α, β]
    end
    R
end


"""
$(TYPEDSIGNATURES)

Calculates the norm of an MPS \$\\ket{\\phi}\$
"""
LinearAlgebra.norm(ψ::AbstractMPS) = sqrt(abs(dot(ψ, ψ)))


"""
$(TYPEDSIGNATURES)

Calculates \$\\bra{\\phi} O \\ket{\\psi}\$

# Details

Calculates the matrix element of \$O\$
```math
\\bra{\\phi} O \\ket{\\psi}
```
in one pass, utlizing `TensorOperations`.
"""

function LinearAlgebra.dot(ϕ::AbstractMPS, O::Union{Vector, NTuple}, ψ::AbstractMPS) #where T <: AbstractMatrix
    S = promote_type(eltype(ψ), eltype(ϕ), eltype(O[1]))
    C = similar(ψ[1], S, (1, 1))
    C[1, 1] = one(S)

    for i ∈ eachindex(ψ)
        M = ψ[i]
        M̃ = conj.(ϕ[i])
        Mat = O[i]
        @tensor C[x, y] := M̃[β, σ, x] * Mat[σ, η] * C[β, α] * M[α, η, y] order = (α, η, β, σ)
    end
    tr(C)
end


function LinearAlgebra.dot(O::AbstractMPO, ψ::AbstractMPS)
    L = length(ψ)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, L)

    for i ∈ 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), σ, (y, b)] := sum(η) W[x, σ, y, η] * M[a, η, b]
        ϕ[i] = N
    end
    ϕ
end

function dot!(ψ::AbstractMPS, O::AbstractMPO)
    L = length(ψ)
    for i ∈ 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), σ, (y, b)] := sum(η) W[x, σ, y, η] * M[a, η, b]
        ψ[i] = N
    end
end

function LinearAlgebra.dot(O1::AbstractMPO, O2::AbstractMPO)
    L = length(O1)
    S = promote_type(eltype(O1), eltype(O2))
    T = typeof(O1)
    O = T.name.wrapper(S, L)

    for i ∈ 1:L
        W1 = O1[i]
        W2 = O2[i]
        @reduce V[(x, a), σ, (y, b), η] := sum(γ) W1[x, σ, y, γ] * W2[a, γ, b, η]

        O[i] = V
    end
    O
end

Base.:(*)(A::AbstractTensorNetwork, B::AbstractTensorNetwork) = dot(A, B)
