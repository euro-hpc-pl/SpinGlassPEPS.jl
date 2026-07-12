# ./mps/dot.jl: This file provides basic functionality to compute the dot product between MPS
#               Other functions to contract MPS with other tensors are also provided.

LinearAlgebra.norm(ψ::QMps) = sqrt(abs(dot(ψ, ψ)))

Base.:(*)(ϕ::QMps, ψ::QMps) = dot(ϕ, ψ)
Base.:(*)(W::QMpo, ψ::QMps) = dot(W, ψ)

function LinearAlgebra.dot(ψ::QMps{T}, ϕ::QMps{T}) where {T<:Real}
    @assert ψ.sites == ϕ.sites
    C = ψ.onGPU && ϕ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for i ∈ ϕ.sites
        A, B = ϕ[i], ψ[i]
        @tensor order = (α, β, σ) C[x, y] := conj(B)[β, x, σ] * C[β, α] * A[α, y, σ]
    end
    tr(C)
end

function LinearAlgebra.dot(ψ::QMpo{R}, ϕ::QMps{R}) where {R<:Real}
    D = TensorMap{R}()
    for i ∈ reverse(ϕ.sites)
        M, B = ψ[i], ϕ[i]
        for v ∈ reverse(M.bot)
            B = contract_matrix_tensor3(v, B)
        end
        B = contract_tensors43(M.ctr, B)
        for v ∈ reverse(M.top)
            B = contract_matrix_tensor3(v, B)
        end

        mps_li = left_nbrs_site(i, ϕ.sites)
        mpo_li = left_nbrs_site(i, ψ.sites)

        while mpo_li > mps_li
            st = size(B, 3)
            sl2 = size(ψ[mpo_li], 2)
            # @cast B[l1, l2, (r, t)] := B[(l1, l2), r, t] (l2 ∈ 1:sl2)
            B = reshape(B, size(B, 1) ÷ sl2, sl2, size(B, 2) * size(B, 3))
            B = permutedims(B, (1, 3, 2))
            B = contract_matrix_tensor3(ψ[mpo_li], B)
            B = permutedims(B, (1, 3, 2))
            # @cast B[(l1, l2), r, t] := B[l1, l2, (r, t)] (t ∈ 1:st)
            B = reshape(B, size(B, 1) * size(B, 2), size(B, 3) ÷ st, st)
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => B)
    end
    QMps(D; onGPU = ψ.onGPU && ϕ.onGPU)
end

contract_tensor3_matrix(B::AbstractArray{T,3}, M::MpoTensor{T,2}) where {T<:Real} =
    contract_tensor3_matrix(B, M.ctr)
contract_matrix_tensor3(M::MpoTensor{T,2}, B::AbstractArray{T,3}) where {T<:Real} =
    contract_matrix_tensor3(M.ctr, B)
contract_tensors43(B::Nothing, A::AbstractArray{T,3}) where {T<:Real} = A
