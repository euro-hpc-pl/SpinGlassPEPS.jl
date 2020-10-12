
function dot(ψ::MPS{T}, O::Vector{AbstractMatrix}, ϕ::MPS{T}) where T <: AbstractArray{3}
    L = length(ϕ)
    C = I

    for i ∈ 1:L
        M = ϕ.tensors[i]
        N = ψ.tensors[i]
        @tensor C[α, β] := N[y, α, σ] * O[i][σ, η] * C[y, x] * M[x, α, σ] order = (x, η, y, σ)
    end
    return C
end

function dot(ψ::MPS{T}, ϕ::MPS{T}) where T <: AbstractArray{3}
    L = length(ϕ)
    O = [I for _ ∈ 1:L]
    return  dot(ψ, O, ϕ)
end

function norm(ψ::MPS{T}) where T <: AbstractArray{3}
    return sqrt(dot(ψ, ψ))
end