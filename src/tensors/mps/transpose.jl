
# ./mps/transpose.jl: This file defines what it means to transpse MPO. Note, this should not be
#                     done by overloading Base.transpose for QMpo to avoid overloading (Array)'.

function Base.transpose(ψ::QMpo{T}) where {T<:Real}
    QMpo(
        MpoTensorMap{T}(keys(ψ.tensors) .=> mpo_transpose.(values(ψ.tensors)));
        onGPU = ψ.onGPU,
    )
end

mpo_transpose(M::MpoTensor{T,2}) where {T<:Real} = M

function mpo_transpose(M::MpoTensor{T,4}) where {T<:Real}
    MpoTensor{T,4}(
        mpo_transpose.(reverse(M.bot)),
        mpo_transpose(M.ctr),
        mpo_transpose.(reverse(M.top)),
        M.dims[[1, 4, 3, 2]],
    )
end
