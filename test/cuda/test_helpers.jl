cuproj(state, dims::Union{Vector, NTuple}) = cu.(proj(state, dims))

function tensor(ψ::CuMPS)
    devψ = MPS(ψ)
    t = tensor(devψ)
    cu(t)
end