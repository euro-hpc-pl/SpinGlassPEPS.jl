# diagonal.jl: contractions with DiagonalTensor on CPU and CUDA

function contract_tensor3_matrix(B::Tensor{R,3}, C::DiagonalTensor{R}) where {R<:Real}
    s2 = size(C.e2, 1)
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3) ÷ s2, s2)
    B = contract_tensor3_matrix(B, C.e2)
    s1 = size(C.e1, 1)
    B = reshape(B, size(B, 1), size(B, 2) ÷ s1, s1, size(B, 3))
    B = permutedims(B, (1, 2, 4, 3))
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3), size(B, 4))
    B = contract_tensor3_matrix(B, C.e1)
    q2 = size(C.e2, 2)
    B = reshape(B, size(B, 1), size(B, 2) ÷ q2, size(B, 3) * q2)
end

function contract_matrix_tensor3(C::DiagonalTensor{R}, B::Tensor{R,3}) where {R<:Real}
    s1 = size(C.e1, 2)
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3) ÷ s1, s1)
    B = contract_matrix_tensor3(C.e1, B)
    s2 = size(C.e2, 2)
    B = reshape(B, size(B, 1), size(B, 2) ÷ s2, s2, size(B, 3))
    B = permutedims(B, (1, 2, 4, 3))
    B = reshape(B, size(B, 1), size(B, 2) * size(B, 3), size(B, 4))
    B = contract_matrix_tensor3(C.e2, B)
    q1 = size(C.e1, 1)
    B = reshape(B, size(B, 1), size(B, 2) ÷ q1, size(B, 3) * q1)
end
