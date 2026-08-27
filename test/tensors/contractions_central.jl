# contractions_central.jl
#
# Unit tests for src/contractions/central.jl (CentralTensor contraction kernels).
#
# Reference convention (derived independently from the defining equation of
# CentralTensor, cf. src/base.jl `dense_central` docstring-comment):
#
#   A CentralTensor M with fields
#       e11 :: (sl1 x sr1), e12 :: (sl1 x sr2),
#       e21 :: (sl2 x sr1), e22 :: (sl2 x sr2)
#   represents the dense matrix
#       E[(l1,l2), (r1,r2)] = e11[l1,r1] * e21[l2,r1] * e12[l1,r2] * e22[l2,r2]
#   with column-major (Julia) index fusion: row L = l1 + (l2-1)*sl1 (l1 fastest),
#   column R = r1 + (r2-1)*sr1 (r1 fastest); size(M) == (sl1*sl2, sr1*sr2).
#   E is materialized here with explicit loops (ref_central_dense) -- NOT with
#   any kernel from src -- and expected results are computed with @tensor.
#
#   Contracts pinned down (call-site contracts from src/contractions/central.jl,
#   src/contractions/dense.jl and src/contractions/virtual.jl):
#     * contract_tensor3_matrix(LE[b,t,L], M)              -> out[b,t,R] = sum_L LE[b,t,L] * E[L,R]
#     * contract_matrix_tensor3(M, RE[b,t,R])              -> out[b,t,L] = sum_R RE[b,t,R] * E[L,R]
#     * SpinGlassTensors.batched_mul!(out[b,R,t], LE[b,L,t], M)  -> out[b,R,t] = sum_L LE[b,L,t] * E[L,R]
#       (same contract for the AbstractMatrix variant of batched_mul!)
#     * update_reduced_env_right(RR[x,R], M)               -> out[x,L] = sum_R RR[x,R] * E[L,R]
#
#   dense_central(M) equals E ./ maximum(E)  (src normalizes by the maximum).
#
# contract_tensor3_central (and the CentralTensor batched_mul!) dispatch among
# FIVE size-heuristic branches based on (sb*st, sl1, sl2, sr1, sr2). The fixture
# list below hits all five branches for the forward call and -- because
# contract_matrix_tensor3 re-enters the kernel with dims (sr1,sr2,sl1,sl2) --
# all five for the adjoint direction as well. This is asserted explicitly.

using Test
using SpinGlassPEPS.SpinGlassTensors
using TensorOperations
using LinearAlgebra
using CUDA

# --- independent dense materialization of a CentralTensor (explicit loops) ---
function ref_central_dense(
    e11::AbstractMatrix{T},
    e12::AbstractMatrix{T},
    e21::AbstractMatrix{T},
    e22::AbstractMatrix{T},
) where {T}
    sl1, sr1 = size(e11)
    sl2, sr2 = size(e22)
    @assert size(e12) == (sl1, sr2) && size(e21) == (sl2, sr1)
    E = zeros(T, sl1 * sl2, sr1 * sr2)
    for r2 ∈ 1:sr2, r1 ∈ 1:sr1, l2 ∈ 1:sl2, l1 ∈ 1:sl1
        E[l1+(l2-1)*sl1, r1+(r2-1)*sr1] =
            e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
    end
    E
end

# --- replica of the branch predicate (for fixture labeling only, not numerics) ---
function central_branch(sbt, sl1, sl2, sr1, sr2)
    sinter = sbt * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        1
    elseif sr1 <= sr2 && sl1 <= sl2
        2
    elseif sr1 <= sr2 && sl2 <= sl1
        3
    elseif sr2 <= sr1 && sl1 <= sl2
        4
    else # sr2 <= sr1 && sl2 <= sl1
        5
    end
end

# (sl1, sl2, sr1, sr2, sb, st); chosen so the forward kernel calls hit
# branches 1..5 (in this order) and the adjoint calls hit {1,2,4,3,5}.
const central_fixtures = [
    (sl1 = 2, sl2 = 2, sr1 = 2, sr2 = 2, sb = 4, st = 5),
    (sl1 = 3, sl2 = 5, sr1 = 3, sr2 = 4, sb = 2, st = 2),
    (sl1 = 5, sl2 = 3, sr1 = 3, sr2 = 4, sb = 2, st = 2),
    (sl1 = 3, sl2 = 5, sr1 = 4, sr2 = 3, sb = 2, st = 2),
    (sl1 = 5, sl2 = 3, sr1 = 4, sr2 = 3, sb = 2, st = 2),
]

@testset "central fixtures cover all five kernel branches" begin
    fwd = [central_branch(f.sb * f.st, f.sl1, f.sl2, f.sr1, f.sr2) for f ∈ central_fixtures]
    # contract_matrix_tensor3 calls the kernel with (e11', e21', e12', e22'),
    # i.e. with effective dims (sr1, sr2, sl1, sl2)
    bwd = [central_branch(f.sb * f.st, f.sr1, f.sr2, f.sl1, f.sl2) for f ∈ central_fixtures]
    @test sort(fwd) == collect(1:5)
    @test sort(bwd) == collect(1:5)
    @test fwd == [1, 2, 3, 4, 5]
end

test_gpu_central = CUDA.functional()

@testset "CentralTensor contraction kernels vs dense reference (T = $T)" for T ∈ (
    Float64,
    Float32,
)
    rtol = 1000 * eps(T)
    for (i, f) ∈ enumerate(central_fixtures)
        sl1, sl2, sr1, sr2, sb, st = f.sl1, f.sl2, f.sr1, f.sr2, f.sb, f.st
        fwd_branch = central_branch(sb * st, sl1, sl2, sr1, sr2)
        bwd_branch = central_branch(sb * st, sr1, sr2, sl1, sl2)

        e11 = rand(T, sl1, sr1)
        e12 = rand(T, sl1, sr2)
        e21 = rand(T, sl2, sr1)
        e22 = rand(T, sl2, sr2)
        M = CentralTensor(e11, e12, e21, e22)
        E = ref_central_dense(e11, e12, e21, e22)

        # CPU inputs
        LE = rand(T, sb, st, sl1 * sl2)   # for contract_tensor3_matrix
        RE = rand(T, sb, st, sr1 * sr2)   # for contract_matrix_tensor3
        LE3 = rand(T, sb, sl1 * sl2, st)  # for batched_mul!
        RR = rand(T, sb, sr1 * sr2)       # for update_reduced_env_right

        # independent references
        @tensor ref_fwd[b, t, r] := LE[b, t, l] * E[l, r]
        @tensor ref_bwd[b, t, l] := RE[b, t, r] * E[l, r]
        @tensor ref_bat[b, r, t] := LE3[b, l, t] * E[l, r]
        @tensor ref_red[x, l] := RR[x, r] * E[l, r]

        @testset "fixture $i (fwd branch $fwd_branch, adj branch $bwd_branch), CPU" begin
            @test size(M) == (sl1 * sl2, sr1 * sr2)
            @test M.dims == (sl1 * sl2, sr1 * sr2)
            @test eltype(M) == T

            # dense_central agrees with our materialization up to its max-normalization
            @test dense_central(M) ≈ E ./ maximum(E) rtol = rtol

            out_fwd = contract_tensor3_matrix(LE, M)
            @test out_fwd isa Array{T,3}
            @test size(out_fwd) == (sb, st, sr1 * sr2)
            @test out_fwd ≈ ref_fwd rtol = rtol

            out_bwd = contract_matrix_tensor3(M, RE)
            @test out_bwd isa Array{T,3}
            @test size(out_bwd) == (sb, st, sl1 * sl2)
            @test out_bwd ≈ ref_bwd rtol = rtol

            # CentralTensor variant of batched_mul! (must overwrite, not accumulate)
            out_bat = fill(T(1) / 2, sb, sr1 * sr2, st)
            SpinGlassTensors.batched_mul!(out_bat, LE3, M)
            @test out_bat ≈ ref_bat rtol = rtol

            # AbstractMatrix variant of batched_mul! (same file), fed our dense E
            out_mat = fill(T(1) / 2, sb, sr1 * sr2, st)
            SpinGlassTensors.batched_mul!(out_mat, LE3, E)
            @test out_mat ≈ ref_bat rtol = rtol

            out_red = update_reduced_env_right(RR, M)
            @test out_red isa Array{T,2}
            @test size(out_red) == (sb, sl1 * sl2)
            @test out_red ≈ ref_red rtol = rtol
        end

        if test_gpu_central
            @testset "fixture $i (fwd branch $fwd_branch, adj branch $bwd_branch), CUDA" begin
                # move_to_CUDA! mutates the fields, so give it its own copy
                Mg = move_to_CUDA!(
                    CentralTensor(copy(e11), copy(e12), copy(e21), copy(e22)),
                )
                @test Mg.e11 isa CuArray && Mg.e22 isa CuArray

                g_fwd = contract_tensor3_matrix(CuArray(LE), Mg)
                @test g_fwd isa CuArray{T,3}
                @test size(g_fwd) == (sb, st, sr1 * sr2)
                @test Array(g_fwd) ≈ ref_fwd rtol = rtol

                g_bwd = contract_matrix_tensor3(Mg, CuArray(RE))
                @test g_bwd isa CuArray{T,3}
                @test size(g_bwd) == (sb, st, sl1 * sl2)
                @test Array(g_bwd) ≈ ref_bwd rtol = rtol

                g_bat = CUDA.fill(T(1) / 2, sb, sr1 * sr2, st)
                SpinGlassTensors.batched_mul!(g_bat, CuArray(LE3), Mg)
                @test Array(g_bat) ≈ ref_bat rtol = rtol

                # AbstractMatrix variant: M given on CPU, moved inside via ArrayorCuArray
                g_mat = CUDA.fill(T(1) / 2, sb, sr1 * sr2, st)
                SpinGlassTensors.batched_mul!(g_mat, CuArray(LE3), E)
                @test Array(g_mat) ≈ ref_bat rtol = rtol

                g_red = update_reduced_env_right(CuArray(RR), Mg)
                @test g_red isa CuArray{T,2}
                @test size(g_red) == (sb, sl1 * sl2)
                @test Array(g_red) ≈ ref_red rtol = rtol
            end
        end
    end
end

# update_reduced_env_right must agree with its dense-matrix counterpart
# (dense.jl: RR[x, y] := M0[y, z] * RR[x, z] with M0 = dense E)
@testset "update_reduced_env_right consistency dense vs CentralTensor (T = $T)" for T ∈ (
    Float64,
    Float32,
)
    rtol = 1000 * eps(T)
    e11, e12, e21, e22 = rand(T, 2, 3), rand(T, 2, 2), rand(T, 3, 3), rand(T, 3, 2)
    M = CentralTensor(e11, e12, e21, e22)
    E = ref_central_dense(e11, e12, e21, e22)
    RR = rand(T, 4, size(E, 2))
    @test update_reduced_env_right(RR, M) ≈ update_reduced_env_right(RR, E) rtol = rtol
end
