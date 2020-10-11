_create_empty(::Type{T}, l) where {T <: CuVector{S}} where {S <: Number} = Vector{CuArray{S, 3}}(undef, l)

function Base.:(*)(ψ′::Adjoint{S, MPS{T}}, ϕ::MPS{T}) where {T <: CuArray{S, 3}} where {S <: Number}
    ψ = ψ′.parent

    M   = ϕ.tensors[1]
    M̃dg = dg(ψ.tensors[1])

    @cutensor cont[b₁, a₁] := M̃dg[b₁, 1, σ₁] * M[1, a₁, σ₁]

    l = length(ϕ)
    for i=2:l-1
        M   = ϕ.tensors[i]
        M̃dg = dg(ψ.tensors[i])

        @cutensor cont[bᵢ, aᵢ] := M̃dg[bᵢ, bᵢ₋₁, σᵢ] * cont[bᵢ₋₁, aᵢ₋₁] * M[aᵢ₋₁, aᵢ, σᵢ]
    end
    M   = ϕ.tensors[l]
    M̃dg = dg(ψ.tensors[l])
    
    @cutensor M̃dg[1, bᴸ⁻¹, σᴸ] * cont[bᴸ⁻¹, aᴸ⁻¹] * M[aᴸ⁻¹, 1, σᴸ]
end