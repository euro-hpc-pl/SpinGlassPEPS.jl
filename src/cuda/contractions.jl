export left_env

function LinearAlgebra.dot(ϕ::CuMPS, ψ::CuMPS)
    C = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ eachindex(ψ)
        M = ψ[i]
        M̃ = conj.(ϕ[i])
        @cutensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
    end
    return tr(C)
end

function left_env(ϕ::CuMPS, ψ::CuMPS)
    l = length(ψ)
    S = promote_type(eltype(ψ), eltype(ϕ))

    L = Vector{CuMatrix{S}}(undef, l+1)
    L[1] = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ 1:l
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        C = L[i]
        @cutensor C[x, y] := M̃[β, σ, x] * C[β, α] * M[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    return L
end

function right_env(ϕ::CuMPS, ψ::CuMPS) 
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))

    R = Vector{CuMatrix{T}}(undef, L+1)
    R[end] = ones(eltype(ψ), 1, 1)

    for i ∈ L:-1:1
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        D = R[i+1]
        @cutensor D[x, y] := M[x, σ, α] * D[α, β] * M̃[y, σ, β] order = (β, α, σ)
        R[i] = D
    end
    return R
end

function LinearAlgebra.dot(ϕ::CuMPS, O::Vector{<:CuMatrix}, ψ::CuMPS)
    S = promote_type(eltype(ψ), eltype(ϕ), eltype(O[1]))
    C = CUDA.ones(S, 1, 1)

    for i ∈ eachindex(ψ)
        M = ψ[i]
        M̃ = conj.(ϕ[i])
        Mat = O[i]
        @cutensor C[x, y] := M̃[β, σ, x] * Mat[σ, η] * C[β, α] * M[α, η, y] order = (α, η, β, σ)
    end
    return tr(C)
end