# Unit tests for src/contractions/diagonal.jl: contract_tensor3_matrix / contract_matrix_tensor3
# with a DiagonalTensor argument, pinned against an INDEPENDENT dense reference.
#
# Reference convention (derived from the column-major reshape chains in
# src/contractions/diagonal.jl together with the dense kernel in src/contractions/dense.jl
# and the central kernel in src/contractions/central.jl):
#
#   * A DiagonalTensor C with blocks e1 (s1 x q1) and e2 (s2 x q2) acts as the dense matrix
#     D of size (s1*s2, q1*q2) with
#
#         D[i1 + (i2-1)*s1, j2 + (j1-1)*q2] = E1[i1, j1] * E2[i2, j2]
#
#     where E1, E2 are the dense matrices of the blocks. Note the ASYMMETRIC fusion:
#     on the input ("row", MPO-left) leg e1's row index i1 is FASTEST, while on the
#     output ("col", MPO-right) leg e2's column index j2 is FASTEST. This convention is
#     self-consistent with mpo_transpose(::DiagonalTensor) in src/base.jl, which maps
#     (e1, e2) -> (mpo_transpose(e2), mpo_transpose(e1)) and equals transpose(D)
#     under the convention above (verified in a testset below).
#
#   * A CentralTensor block with e11 (l1 x r1), e12 (l1 x r2), e21 (l2 x r1), e22 (l2 x r2)
#     acts as the dense matrix E of size (l1*l2, r1*r2), both legs fused "1-fastest":
#
#         E[u1 + (u2-1)*l1, d1 + (d2-1)*r1] = e11[u1,d1] * e12[u1,d2] * e21[u2,d1] * e22[u2,d2]
#
#     (this is dense_central from src/base.jl WITHOUT its normalization by maximum(V)).
#
#   * Kernels under test (B is an MPS-like 3-tensor, leg 3 is the MPO leg):
#         contract_tensor3_matrix(B, C):  out[a, b, r] = sum_l B[a, b, l] * D[l, r]
#         contract_matrix_tensor3(C, B):  out[a, b, l] = sum_r B[a, b, r] * D[l, r]
#
# The dense references below are built with explicit loops / plain @tensor and never call
# the kernels under test.

using Test
using LinearAlgebra
using TensorOperations
using CUDA
using Random
using SpinGlassTensors

# --- independent dense materialization -------------------------------------------------

ref_dense_block(M::AbstractMatrix) = Matrix(M)

function ref_dense_block(M::CentralTensor)
    l1, r1 = size(M.e11)
    l2, r2 = size(M.e22)
    E = zeros(eltype(M.e11), l1 * l2, r1 * r2)
    for u1 = 1:l1, u2 = 1:l2, d1 = 1:r1, d2 = 1:r2
        E[u1+(u2-1)*l1, d1+(d2-1)*r1] =
            M.e11[u1, d1] * M.e12[u1, d2] * M.e21[u2, d1] * M.e22[u2, d2]
    end
    E
end

function ref_dense_diagonal(C::DiagonalTensor)
    E1 = ref_dense_block(C.e1)
    E2 = ref_dense_block(C.e2)
    s1, q1 = size(E1)
    s2, q2 = size(E2)
    D = zeros(promote_type(eltype(E1), eltype(E2)), s1 * s2, q1 * q2)
    for i1 = 1:s1, i2 = 1:s2, j1 = 1:q1, j2 = 1:q2
        D[i1+(i2-1)*s1, j2+(j1-1)*q2] = E1[i1, j1] * E2[i2, j2]
    end
    D
end

function ref_tensor3_matrix(B::Array{R,3}, D::Matrix{R}) where {R<:Real}
    @tensor out[a, b, r] := B[a, b, l] * D[l, r]
    out
end

function ref_matrix_tensor3(D::Matrix{R}, B::Array{R,3}) where {R<:Real}
    @tensor out[a, b, l] := B[a, b, r] * D[l, r]
    out
end

# --- fixtures ---------------------------------------------------------------------------

rtol_for(::Type{Float64}) = 1e-12
rtol_for(::Type{Float32}) = 1e-5

rand_signed(::Type{R}, dims...) where {R<:Real} = rand(R, dims...) .- R(1 // 2)

function diag_dense_blocks(::Type{R}) where {R<:Real}
    # all four block dimensions distinct to catch index-order mistakes
    DiagonalTensor(rand_signed(R, 2, 4), rand_signed(R, 3, 5))
end

function central_block(::Type{R}, l1, l2, r1, r2) where {R<:Real}
    CentralTensor(
        rand_signed(R, l1, r1),
        rand_signed(R, l1, r2),
        rand_signed(R, l2, r1),
        rand_signed(R, l2, r2),
    )
end

function diag_central_blocks(::Type{R}) where {R<:Real}
    # mirrors the production construction in
    # lib/SpinGlassEngine/src/square_cross_double_node.jl (:central_d_double_node):
    # DiagonalTensor(CentralTensor, CentralTensor)
    DiagonalTensor(central_block(R, 2, 3, 3, 2), central_block(R, 2, 2, 2, 3))
end

function diag_mixed_blocks(::Type{R}) where {R<:Real}
    DiagonalTensor(rand_signed(R, 3, 2), central_block(R, 2, 2, 2, 3))
end

fixture_families(::Type{R}) where {R<:Real} = (
    "dense blocks" => diag_dense_blocks(R),
    "central blocks" => diag_central_blocks(R),
    "mixed blocks" => diag_mixed_blocks(R),
)

# --- tests ------------------------------------------------------------------------------

Random.seed!(1234)

@testset "DiagonalTensor dims and reference materialization agree" begin
    for R ∈ (Float64, Float32), (name, C) ∈ fixture_families(R)
        D = ref_dense_diagonal(C)
        @test size(C) == size(D)
        @test size(C, 1) == size(ref_dense_block(C.e1), 1) * size(ref_dense_block(C.e2), 1)
        @test size(C, 2) == size(ref_dense_block(C.e1), 2) * size(ref_dense_block(C.e2), 2)
        @test eltype(D) == R
    end
end

@testset "contract_tensor3_matrix(B, ::DiagonalTensor) vs dense reference ($R, $name)" for R ∈
                                                                                           (
        Float64,
        Float32,
    ),
    (name, C) ∈ fixture_families(R)

    D = ref_dense_diagonal(C)
    B = rand_signed(R, 3, 2, size(C, 1))
    out = contract_tensor3_matrix(B, C)
    expected = ref_tensor3_matrix(B, D)
    @test out isa Array{R,3}
    @test size(out) == (3, 2, size(C, 2))
    @test out ≈ expected rtol = rtol_for(R)
end

@testset "contract_matrix_tensor3(::DiagonalTensor, B) vs dense reference ($R, $name)" for R ∈
                                                                                           (
        Float64,
        Float32,
    ),
    (name, C) ∈ fixture_families(R)

    D = ref_dense_diagonal(C)
    B = rand_signed(R, 3, 2, size(C, 2))
    out = contract_matrix_tensor3(C, B)
    expected = ref_matrix_tensor3(D, B)
    @test out isa Array{R,3}
    @test size(out) == (3, 2, size(C, 1))
    @test out ≈ expected rtol = rtol_for(R)
end

@testset "mpo_transpose(::DiagonalTensor) is the matrix transpose of the dense form ($R, $name)" for R ∈
                                                                                                     (
        Float64,
        Float32,
    ),
    (name, C) ∈ fixture_families(R)

    D = ref_dense_diagonal(C)
    Ct = SpinGlassTensors.mpo_transpose(C)
    @test size(Ct) == (size(C, 2), size(C, 1))
    # convention check: src's mpo_transpose must equal transpose of our dense reference
    @test ref_dense_diagonal(Ct) ≈ Matrix(transpose(D)) rtol = rtol_for(R)

    # kernel-level consistency: contracting with the transposed tensor from the left
    # equals contracting with the original tensor from the right
    B = rand_signed(R, 2, 3, size(C, 2))
    @test contract_tensor3_matrix(B, Ct) ≈ contract_matrix_tensor3(C, B) rtol = rtol_for(R)
end

@testset "trivial identity blocks act as identity" begin
    for R ∈ (Float64, Float32)
        C = DiagonalTensor(Matrix{R}(I, 3, 3), Matrix{R}(I, 2, 2))
        D = ref_dense_diagonal(C)
        B = rand_signed(R, 2, 2, 6)
        # D is a permutation of the identity under the asymmetric fusion convention;
        # both kernel and reference must agree on it exactly
        @test contract_tensor3_matrix(B, C) ≈ ref_tensor3_matrix(B, D) rtol = rtol_for(R)
        @test contract_matrix_tensor3(C, B) ≈ ref_matrix_tensor3(D, B) rtol = rtol_for(R)
        # and applying the transpose undoes it
        Ct = SpinGlassTensors.mpo_transpose(C)
        @test contract_tensor3_matrix(contract_tensor3_matrix(B, C), Ct) ≈ B rtol =
            rtol_for(R)
    end
end

if CUDA.functional()
    @testset "CUDA: DiagonalTensor kernels vs CPU dense reference ($R, $name)" for R ∈ (
            Float64,
            Float32,
        ),
        (name, C) ∈ fixture_families(R)

        D = ref_dense_diagonal(C)
        Cg = SpinGlassTensors.move_to_CUDA!(deepcopy(C))

        B1 = rand_signed(R, 3, 2, size(C, 1))
        out1 = contract_tensor3_matrix(CuArray(B1), Cg)
        @test out1 isa CuArray{R,3}
        @test Array(out1) ≈ ref_tensor3_matrix(B1, D) rtol = rtol_for(R)

        B2 = rand_signed(R, 3, 2, size(C, 2))
        out2 = contract_matrix_tensor3(Cg, CuArray(B2))
        @test out2 isa CuArray{R,3}
        @test Array(out2) ≈ ref_matrix_tensor3(D, B2) rtol = rtol_for(R)
    end
else
    @info "CUDA not functional; skipping GPU tests in contractions_diagonal.jl"
end
