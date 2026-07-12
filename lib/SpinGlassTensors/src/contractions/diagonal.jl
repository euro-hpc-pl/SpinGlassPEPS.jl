# diagonal.jl: contractions with DiagonalTensor on CPU and CUDA

function contract_tensor3_matrix(B::Tensor{R,3}, C::DiagonalTensor{R}) where {R<:Real}
    # @cast B[l, (r, s1), s2] := B[l, r, (s1, s2)] (s2 ∈ 1:size(C.e2, 1))
    s2 = size(C.e2, 1)
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3) ÷ s2, s2)
    B = contract_tensor3_matrix(B, C.e2)
    # @cast B[l, r, s1, q2] := B[l, (r, s1), q2] (s1 ∈ 1:size(C.e1, 1))
    s1 = size(C.e1, 1)
    B = reshape(B, size(B, 1), size(B, 2) ÷ s1, s1, size(B, 3))
    B = permutedims(B, (1, 2, 4, 3))
    # @cast B[l, (r, q2), s1] := B[l, r, q2, s1]
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3), size(B, 4))
    B = contract_tensor3_matrix(B, C.e1)
    # @cast B[l, r, (q2, q1)] := B[l, (r, q2), q1] (q2 ∈ 1:size(C.e2, 2))
    q2 = size(C.e2, 2)
    B = reshape(B, size(B, 1), size(B, 2) ÷ q2, size(B, 3) * q2)
end

function contract_matrix_tensor3(C::DiagonalTensor{R}, B::Tensor{R,3}) where {R<:Real}
    # @cast B[l, (r, s2), s1] := B[l, r, (s2, s1)] (s1 ∈ 1:size(C.e1, 2))
    s1 = size(C.e1, 2)
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3) ÷ s1, s1)
    B = contract_matrix_tensor3(C.e1, B)
    # @cast B[l, r, s2, q1] := B[l, (r, s2), q1] (s2 ∈ 1:size(C.e2, 2))
    s2 = size(C.e2, 2)
    B = reshape(B, size(B, 1), size(B, 2) ÷ s2, s2, size(B, 3))
    B = permutedims(B, (1, 2, 4, 3))
    # @cast B[l, (r, q1), s2] := B[l, r, q1, s2]
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3), size(B, 4))
    B = contract_matrix_tensor3(C.e2, B)
    # @cast B[l, r, (q1, q2)] := B[l, (r, q1), q2] (q1 ∈ 1:size(C.e1, 1))
    q1 = size(C.e1, 1)
    B = reshape(B, size(B, 1), size(B, 2) ÷ q1, size(B, 3) * q1)
end
