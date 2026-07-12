# contractions with CentralTensor on CPU and CUDA

export contract_tensor3_matrix, contract_matrix_tensor3, update_reduced_env_right
# my_batched_mul!

function contract_tensor3_matrix(LE::Tensor{R,3}, M::CentralTensor{R,2}) where {R<:Real}
    onGPU = typeof(LE) <: CuArray ? true : false
    contract_tensor3_central(LE, M.e11, M.e12, M.e21, M.e22, onGPU)
end

function contract_matrix_tensor3(M::CentralTensor{R,2}, RE::Tensor{R,3}) where {R<:Real}
    onGPU = typeof(RE) <: CuArray ? true : false
    contract_tensor3_central(RE, M.e11', M.e21', M.e12', M.e22', onGPU)
end

function update_reduced_env_right(RR::Tensor{R,2}, M::CentralTensor{R,2}) where {R<:Real}
    RR = reshape(RR, size(RR, 1), 1, size(RR, 2))
    dropdims(contract_matrix_tensor3(M, RR), dims = 2)
end


function contract_tensor3_central(LE, e11, e12, e21, e22, onGPU)
    sb, st = size(LE)
    sbt = sb * st
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e22, 1), size(e11, 2), size(e22, 2)
    sinter = sbt * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        # @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        # TODO: terrible hack, rmeove when TensorCast is updated
        a11 = reshape(ArrayorCuArray(e11, onGPU), size(e11, 1), :, size(e11, 2))
        a21 = reshape(ArrayorCuArray(e21, onGPU), :, size(e21, 1), size(e21, 2))
        a12 = reshape(ArrayorCuArray(e12, onGPU), size(e12, 1), 1, 1, size(e12, 2))
        a22 = reshape(ArrayorCuArray(e22, onGPU), 1, size(e22, 1), 1, size(e22, 2))
        E = @__dot__(a11 * a21 * a12 * a22)
        E = reshape(E, size(E, 1) * size(E, 2), size(E, 3) * size(E, 4))
        return reshape(reshape(LE, (sbt, sl1 * sl2)) * E, (sb, st, sr1 * sr2))
    elseif sr1 <= sr2 && sl1 <= sl2
        LE = reshape(LE, sbt, sl1, 1, sl2) .* reshape(e21', 1, 1, sr1, sl2)  # [tb, l1, r1, l2]
        LE = reshape(reshape(LE, sbt * sl1 * sr1, sl2) * e22, (sbt, sl1, sr1, sr2))  # [tb, l1, r1, r2]
        LE .*= reshape(e11, 1, sl1, sr1, 1)  # [tb, l1, r1, r2] .* [:, l1, r1, :]
        LE .*= reshape(e12, 1, sl1, 1, sr2)  # [tb, l1, r1, r2] .* [:, l1, :, r2]
        LE = sum(LE, dims = 2)
    elseif sr1 <= sr2 && sl2 <= sl1
        LE = permutedims(reshape(LE, (sbt, sl1, sl2)), (1, 3, 2))  # [tb, l2, l1]
        LE = reshape(LE, sbt, sl2, 1, sl1) .* reshape(e11', 1, 1, sr1, sl1)  # [tb, l2, r1, l1]
        LE = reshape(reshape(LE, sbt * sl2 * sr1, sl1) * e12, (sbt, sl2, sr1, sr2))  # [tb, l2, r1, r2]
        LE .*= reshape(e21, 1, sl2, sr1, 1)  # [tb, l2, r1, r2] .* [:, l2, r1, :]
        LE .*= reshape(e22, 1, sl2, 1, sr2)  # [tb, l2, r1, r2] .* [:, l2, :, r2]
        LE = sum(LE, dims = 2)
    elseif sr2 <= sr1 && sl1 <= sl2
        LE = reshape(LE, sbt, sl1, 1, sl2) .* reshape(e22', 1, 1, sr2, sl2)  # [tb, l1, r2, l2]
        LE = reshape(reshape(LE, sbt * sl1 * sr2, sl2) * e21, (sbt, sl1, sr2, sr1))  # [tb, l1, r2, r1]
        LE .*= reshape(e11, 1, sl1, 1, sr1)  # [tb, l1, r2, r1] .* [:, l1, :, r1]
        LE .*= reshape(e12, 1, sl1, sr2, 1)  # [tb, l1, r2, r1] .* [:, l1, r2, :]
        LE = permutedims(dropdims(sum(LE, dims = 2), dims = 2), (1, 3, 2))
    else # sr2 <= sr1 && sl2 <= sl1
        LE = permutedims(reshape(LE, (sbt, sl1, sl2)), (1, 3, 2))  # [tb, l2, l1]
        LE = reshape(LE, sbt, sl2, 1, sl1) .* reshape(e12', 1, 1, sr2, sl1)  # [tb, l2, r2, l1]
        LE = reshape(reshape(LE, sbt * sl2 * sr2, sl1) * e11, (sbt, sl2, sr2, sr1))  # [tb, l2, r2, r1]
        LE .*= reshape(e21, 1, sl2, 1, sr1)  # [tb, l2, r2, r1] .* [:, l2, :, r1]
        LE .*= reshape(e22, 1, sl2, sr2, 1)  # [tb, l2, r2, r1] .* [:, l2, r2, :]
        LE = permutedims(dropdims(sum(LE, dims = 2), dims = 2), (1, 3, 2))
    end
    reshape(LE, sb, st, sr1 * sr2)
end

function batched_mul!(
    newLE::Tensor{R,3},
    LE::Tensor{R,3},
    M::AbstractArray{R,2},
) where {R<:Real}
    onGPU = typeof(newLE) <: CuArray ? true : false
    N1, N2 = size(M)
    new_M = ArrayorCuArray(M, onGPU)  # TODO: this is a hack to solve problem with types;
    new_M = reshape(new_M, (N1, N2, 1))
    NNlib.batched_mul!(newLE, LE, new_M)
end

function batched_mul!(
    newLE::Tensor{R,3},
    LE::Tensor{R,3},
    M::CentralTensor{R,2},
) where {R<:Real}
    onGPU = typeof(newLE) <: CuArray ? true : false
    sb, _, st = size(LE)
    sl1, sl2, sr1, sr2 = size(M.e11, 1), size(M.e22, 1), size(M.e11, 2), size(M.e22, 2)
    sinter = sb * st * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        # @cast E[(l1, l2), (r1, r2)] :=
        # M.e11[l1, r1] * M.e21[l2, r1] * M.e12[l1, r2] * M.e22[l2, r2]
        a11 = reshape(ArrayorCuArray(M.e11, onGPU), size(M.e11, 1), :, size(M.e11, 2))
        a21 = reshape(ArrayorCuArray(M.e21, onGPU), :, size(M.e21, 1), size(M.e21, 2))
        a12 = reshape(ArrayorCuArray(M.e12, onGPU), size(M.e12, 1), 1, 1, size(M.e12, 2))
        a22 = reshape(ArrayorCuArray(M.e22, onGPU), 1, size(M.e22, 1), 1, size(M.e22, 2))
        E = @__dot__(a11 * a21 * a12 * a22)
        E = reshape(E, size(E, 1) * size(E, 2), size(E, 3) * size(E, 4))
        E = reshape(E, (sl1 * sl2, sr1 * sr2, 1))
        NNlib.batched_mul!(newLE, LE, E)
    elseif sr1 <= sr2 && sl1 <= sl2
        LE = reshape(LE, sb * sl1, 1, sl2, st) .* reshape(M.e21', 1, sr1, sl2, 1)  # [b * l1, r1, l2, t]
        LE = batched_mul(reshape(LE, sb * sl1 * sr1, sl2, st), M.e22)  # [(b, l1, r1), r2, t]
        LE = reshape(LE, (sb, sl1, sr1, sr2, st))  # [b, l1, r1, r2, t]
        LE .*= reshape(M.e11, 1, sl1, sr1, 1, 1)  # [b, l1, r1, r2, t] .* [:, l1, r1, :, :]
        LE .*= reshape(M.e12, 1, sl1, 1, sr2, 1)  # [b, l1, r1, r2, t] .* [:, l1, :, r2, :]
        sum!(reshape(newLE, (sb, 1, sr1, sr2, st)), LE)
    elseif sr1 <= sr2 && sl2 <= sl1
        LE = reshape(LE, sb, 1, sl1, sl2 * st) .* reshape(M.e11', 1, sr1, sl1, 1)  # [b, r1, l1, l2, t]
        LE = batched_mul(reshape(LE, sb * sr1, sl1, sl2 * st), M.e12)  # [(b, r1), r2, (l2, t)]
        LE = reshape(LE, (sb, sr1, sr2, sl2, st))  # [b, r1, r2, l2, t]
        LE .*= reshape(M.e21', 1, sr1, 1, sl2, 1)  # [b, r1, r2, l2, t] .* [:, r1, :, l2, :]
        LE .*= reshape(M.e22', 1, 1, sr2, sl2, 1)  # [b, r1, r2, l2, t] .* [:, :, r2, l2, :]
        sum!(reshape(newLE, (sb, sr1, sr2, 1, st)), LE)
    elseif sr2 <= sr1 && sl1 <= sl2
        LE = reshape(LE, sb * sl1, sl2, 1, st) .* reshape(M.e22, 1, sl2, sr2, 1)  # [b, l1, l2, r2, t]
        LE = batched_mul(reshape(LE, sb * sl1, sl2, sr2 * st), M.e21) # [(b, l1), r1, (r2, t)]
        LE = reshape(LE, (sb, sl1, sr1, sr2, st))  # [b, l1, r1, r2, t]
        LE .*= reshape(M.e11, 1, sl1, sr1, 1, 1)  # [b, l1, r1, r2, t] .* [:, l1, r1, :, :]
        LE .*= reshape(M.e12, 1, sl1, 1, sr2, 1)  # [b, l1, r1, r2, t] .* [:, l1, :, r2, :]
        sum!(reshape(newLE, (sb, 1, sr1, sr2, st)), LE)
    else # sr2 <= sr1 && sl2 <= sl1
        LE = reshape(LE, sb, sl1, sl2, 1, st) .* reshape(M.e12, 1, sl1, 1, sr2, 1)  # [b, l1, l2, r2, t]
        LE = batched_mul(reshape(LE, sb, sl1, sl2 * sr2 * st), M.e11)  # [b, r1, (l2, r2, t)]
        LE = reshape(LE, (sb, sr1, sl2, sr2, st))  # [b, r1, l2, r2, t]
        LE .*= reshape(M.e21', 1, sr1, sl2, 1, 1)  # [b, r1, l2, r2, t] .* [:, l2, :, r1]
        LE .*= reshape(M.e22, 1, 1, sl2, sr2, 1)  # [b, r1, l2, r2, t] .* [:, :, l2, r2, :]
        sum!(reshape(newLE, (sb, sr1, 1, sr2, st)), LE)
    end
end
