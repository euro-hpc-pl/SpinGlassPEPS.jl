using Base 
export left_env, right_env, dot!

# --------------------------- Conventions ------------------------ 
#                                                                 
#      MPS          MPS*         MPO       left env      left env
#       1            1            1           - 0          1 -
#   0 - A - 2    0 - B - 2    0 - W - 2      L               R
#                                 3           - 1          0 -
# ---------------------------------------------------------------
#
# TODO
# 1) right moving dot version
# 2) write right_env analogous to left_env
# 3) combine 1-2 into one function
# 4) Add more general dots and envs (MPS, MPO, MPS)

function LinearAlgebra.dot(ϕ::MPS, ψ::MPS)
    T = promote_type(eltype(ψ), eltype(ϕ))
    C = ones(T, 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ) 
    end
    return C[1]
end

function left_env(ϕ::MPS, ψ::MPS) 
    l = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))

    L = Vector{Matrix{T}}(undef, l+1)
    L[1] = ones(eltype(ψ), 1, 1)

    for i ∈ 1:l
        M = ψ[i]
        M̃ = conj(ϕ[i])

        C = L[i]
        @tensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    return L
end

# NOT tested yet
function right_env(ϕ::MPS, ψ::MPS) 
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))

    R = Vector{Matrix{T}}(undef, L+1)
    R[end] = ones(eltype(ψ), 1, 1)

    for i ∈ L:-1:1
        M = ψ[i]
        M̃ = conj(ϕ[i])

        D = R[i+1]
        @tensor D[x, y] := M[x, σ, α] * D[α, β] * M̃[y, σ, β] order = (β, α, σ)
        R[i] = D
    end
    return R
end

LinearAlgebra.norm(ψ::MPS) = sqrt(abs(dot(ψ, ψ)))

function LinearAlgebra.dot(ϕ::MPS, O::Vector{T}, ψ::MPS) where {T <: AbstractMatrix}
    S = promote_type(eltype(ψ), eltype(ϕ), eltype(O[1]))
    C = ones(S, 1, 1)

    for i ∈ 1:length(ψ)
        M = ψ[i]
        M̃ = conj(ϕ[i])
        Mat = O[i]
        @tensor C[x, y] := M̃[β, σ, x] * Mat[σ, η] * C[β, α] * M[α, η, y] order = (α, η, β, σ)
end
    return C[1]
end

function LinearAlgebra.dot(O::MPO, ψ::MPS)
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))
    ϕ = MPS(T, L)

    for i in 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), (y, b), σ] := sum(η) W[x, σ, y, η] * M[a, η, b]      
        ϕ[i] = N
    end
    return ϕ
end

function dot!(O::MPO, ψ::MPS)
    L = length(ψ)
    for i in 1:L
        W = O[i]
        M = ψ[i]

        @reduce N[(x, a), (y, b), σ] := sum(η) W[x, σ, y, η] * M[a, η, b]      
        ψ[i] = N
    end
end

Base.:(*)(O::MPO, ψ::MPS) = return dot(O, ψ)

function LinearAlgebra.dot(O1::MPO, O2::MPO)
    L = length(O1)
    T = promote_type(eltype(ψ), eltype(ϕ))
    O = MPO(T, L)

    for i in 1:L
        W1 = O1[i]
        W2 = O2[i]
        
        @reduce V[(x, a), σ, (y, b), η] := sum(γ) W1[x, σ, y, γ] * W2[a, γ, b, η]        
        O[i] = V
    end
    return O
end

Base.:(*)(O1::MPO, O2::MPO) = dot(O1, O2)
