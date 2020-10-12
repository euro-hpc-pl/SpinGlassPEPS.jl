function compress(ψ::MPS{L, T}, to_the::Right; Dcut::Int=typemax(Int)) where {L, T}
    tensors = Array{T, 3}[]
    
    B = ψ[1]
    d = length(B[1, 1, :])
    
    @cast Bm[(σ¹, a⁰), a¹] |= B[a⁰, a¹, σ¹]
    U, S, V = psvd(Bm, rank=Dcut)
    #S = S/√sum(S .^ 2)

    @cast A[a⁰, a¹, σ¹] |= U[(σ¹, a⁰), a¹] (σ¹:d)
    push!(tensors, A)
    
    for i ∈ 2:L
        B = ψ[i]
        d = length(B[1, 1, :])

        @tensor M[aⁱ⁻¹, aⁱ, σⁱ] := (Diagonal(S)*V')[aⁱ⁻¹, aⁱ⁻¹′] * B[aⁱ⁻¹′, aⁱ, σⁱ]
        @cast   Mm[(σⁱ, aⁱ⁻¹), aⁱ] |= M[aⁱ⁻¹, aⁱ, σⁱ]
        
        U, S, V = psvd(Mm, rank=Dcut)
        #S = S/√sum(S .^ 2)

        @cast A[aⁱ⁻¹, aⁱ, σⁱ] |= U[(σⁱ, aⁱ⁻¹), aⁱ] (σⁱ:d)
        push!(tensors, A)
    end
    MPS{L, T}(tensors), Left()
end