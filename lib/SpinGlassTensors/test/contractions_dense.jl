# test/contractions_dense.jl
#
# Unit tests for the dense-argument kernels in src/contractions/dense.jl:
#
#   contract_tensor3_matrix, contract_matrix_tensor3,
#   update_env_left   (arities: (LE3,A3,M4,B3), (LE2,A3,B3), (LE2,A3)),
#   update_env_right  (arities: (RE3,A3,M4,B3), (RE2,A3,B3), (RE3,C3)),
#   project_ket_on_bra (arities: (LE3,B3,M4,RE3), (LE2,B3,RE2), (LE2,RE3)),
#   update_reduced_env_right ((RE2,m,MpoTensor4,B3), (K1,RE2,M4,B3), (RR2,M02)),
#   contract_tensors43, corner_matrix
#
# Reference convention (index order derived from the @tensor bodies / docstrings
# in src/contractions/dense.jl; legs named as in that file):
#
#   contract_tensor3_matrix(A, M)[l, r, y]        = Σ_σ A[l, r, σ] M[σ, y]
#   contract_matrix_tensor3(M, A)[l, r, y]        = Σ_σ A[l, r, σ] M[y, σ]          (A * M')
#   update_env_left(LE, A, M, B)[nb, nt, nc]      = Σ LE[ob,ot,oc] A[ot,nt,α] M[oc,α,nc,β] B[ob,nb,β]
#   update_env_left(LE, A, B)[nb, nt]             = Σ LE[ob,ot] A[ot,nt,α] B[ob,nb,α]
#   update_env_left(LE, A)[nb, nt, nc]            = Σ LE[nb,ot] A[ot,nt,nc]
#   update_env_right(RE, A, M, B)[nb, nt, nc]     = Σ RE[ob,ot,oc] A[nt,ot,α] M[nc,α,oc,β] B[nb,ob,β]
#   update_env_right(RE, A, B)[nb, nt]            = Σ RE[ob,ot] A[nt,ot,α] B[nb,ob,α]
#   update_env_right(RE, C)[nb, nt]               = Σ RE[nb,ot,oc] C[nt,ot,oc]
#   project_ket_on_bra(LE, B, M, RE)[nl, nr, nc]  = Σ LE[ol,nl,lc] B[ol,or,oc] M[lc,nc,rc,oc] RE[or,nr,rc]
#   project_ket_on_bra(LE, B, RE)[nl, nr, nc]     = Σ LE[ol,nl] B[ol,or,nc] RE[or,nr]
#   project_ket_on_bra(LE, RE)[nl, nr, nc]        = Σ LE[ol,nl] RE[ol,nr,nc]
#   update_reduced_env_right(K, RE, M, B)[x, y]   = Σ K[d] M[y,d,β,γ] B[x,α,γ] RE[α,β]
#   update_reduced_env_right(RE, m, Mpo, B)       = the above with K = e_m' * (Mpo.top chain)
#                                                   and B pre-multiplied on leg 3 by the Mpo.bot chain
#                                                   (bot applied as prod(Mpo.bot) acting M-major: B3' = (bot1*bot2*…)[z,σ] B[:,:,σ])
#   update_reduced_env_right(RR, M0)[x, y]        = Σ_z M0[y,z] RR[x,z]
#   contract_tensors43(B, A)[x+(y-1)X, b+(a-1)Bb, z] = Σ_σ B[y,z,a,σ] A[x,b,σ]      (column-major fusing (x,y), (b,a))
#   corner_matrix(C, M, B)[ll, ml, tt, mt]        = Σ M[ml,mt,mr,mb] B[ll,rr,mb] C[rr,tt,mr]
#
# All references below are implemented with pure-Base nested loops -- no
# TensorOperations, no reshapes shared with the kernels -- so they are fully
# independent of the code under test. GPU (CuArray) variants are gated behind
# CUDA.functional() and compared against the same CPU loop references.

using Test
using SpinGlassTensors
using LinearAlgebra
using CUDA
using Random

# not all kernels are exported from SpinGlassTensors
import SpinGlassTensors: update_env_left, update_env_right, project_ket_on_bra, contract_tensors43

# ---------------------------------------------------------------------------
# shared fixture helpers (other test files may copy these)
# ---------------------------------------------------------------------------

dense_test_types() = (Float64, Float32)
dense_test_rtol(::Type{T}) where {T} = √(eps(real(T)))
dense_test_devices() = CUDA.functional() ? (:CPU, :GPU) : (:CPU,)

dense_on(x::AbstractArray, dev::Symbol) = dev == :GPU ? CuArray(x) : x
dense_on(x::Diagonal, dev::Symbol) = dev == :GPU ? Diagonal(CuArray(diag(x))) : x

# ---------------------------------------------------------------------------
# independent loop-based references
# ---------------------------------------------------------------------------

function ref_contract_tensor3_matrix(A::Array{T,3}, M::AbstractMatrix) where {T}
    out = zeros(T, size(A, 1), size(A, 2), size(M, 2))
    for y ∈ axes(M, 2), σ ∈ axes(A, 3), r ∈ axes(A, 2), l ∈ axes(A, 1)
        out[l, r, y] += A[l, r, σ] * M[σ, y]
    end
    out
end

function ref_contract_matrix_tensor3(M::AbstractMatrix, A::Array{T,3}) where {T}
    out = zeros(T, size(A, 1), size(A, 2), size(M, 1))
    for y ∈ axes(M, 1), σ ∈ axes(A, 3), r ∈ axes(A, 2), l ∈ axes(A, 1)
        out[l, r, y] += A[l, r, σ] * M[y, σ]
    end
    out
end

function ref_update_env_left(LE::Array{T,3}, A::Array{T,3}, M::Array{T,4}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 2), size(A, 2), size(M, 3))
    for ob ∈ axes(LE, 1), ot ∈ axes(LE, 2), oc ∈ axes(LE, 3),
        nt ∈ axes(A, 2), α ∈ axes(A, 3), nc ∈ axes(M, 3), β ∈ axes(M, 4), nb ∈ axes(B, 2)

        out[nb, nt, nc] += LE[ob, ot, oc] * A[ot, nt, α] * M[oc, α, nc, β] * B[ob, nb, β]
    end
    out
end

function ref_update_env_left(LE::Array{T,2}, A::Array{T,3}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 2), size(A, 2))
    for ob ∈ axes(LE, 1), ot ∈ axes(LE, 2), nt ∈ axes(A, 2), α ∈ axes(A, 3), nb ∈ axes(B, 2)
        out[nb, nt] += LE[ob, ot] * A[ot, nt, α] * B[ob, nb, α]
    end
    out
end

function ref_update_env_left(LE::Array{T,2}, A::Array{T,3}) where {T}
    out = zeros(T, size(LE, 1), size(A, 2), size(A, 3))
    for nb ∈ axes(LE, 1), ot ∈ axes(LE, 2), nt ∈ axes(A, 2), nc ∈ axes(A, 3)
        out[nb, nt, nc] += LE[nb, ot] * A[ot, nt, nc]
    end
    out
end

function ref_update_env_right(RE::Array{T,3}, A::Array{T,3}, M::Array{T,4}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 1), size(A, 1), size(M, 1))
    for ob ∈ axes(RE, 1), ot ∈ axes(RE, 2), oc ∈ axes(RE, 3),
        nt ∈ axes(A, 1), α ∈ axes(A, 3), nc ∈ axes(M, 1), β ∈ axes(M, 4), nb ∈ axes(B, 1)

        out[nb, nt, nc] += RE[ob, ot, oc] * A[nt, ot, α] * M[nc, α, oc, β] * B[nb, ob, β]
    end
    out
end

function ref_update_env_right(RE::Array{T,2}, A::Array{T,3}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 1), size(A, 1))
    for ob ∈ axes(RE, 1), ot ∈ axes(RE, 2), nt ∈ axes(A, 1), α ∈ axes(A, 3), nb ∈ axes(B, 1)
        out[nb, nt] += RE[ob, ot] * A[nt, ot, α] * B[nb, ob, α]
    end
    out
end

function ref_update_env_right(RE::Array{T,3}, C::Array{T,3}) where {T}
    out = zeros(T, size(RE, 1), size(C, 1))
    for nb ∈ axes(RE, 1), ot ∈ axes(RE, 2), oc ∈ axes(RE, 3), nt ∈ axes(C, 1)
        out[nb, nt] += RE[nb, ot, oc] * C[nt, ot, oc]
    end
    out
end

function ref_project_ket_on_bra(LE::Array{T,3}, B::Array{T,3}, M::Array{T,4}, RE::Array{T,3}) where {T}
    out = zeros(T, size(LE, 2), size(RE, 2), size(M, 2))
    for ol ∈ axes(LE, 1), nl ∈ axes(LE, 2), lc ∈ axes(LE, 3),
        or ∈ axes(B, 2), oc ∈ axes(B, 3), nc ∈ axes(M, 2), rc ∈ axes(M, 3), nr ∈ axes(RE, 2)

        out[nl, nr, nc] += LE[ol, nl, lc] * B[ol, or, oc] * M[lc, nc, rc, oc] * RE[or, nr, rc]
    end
    out
end

function ref_project_ket_on_bra(LE::Array{T,2}, B::Array{T,3}, RE::Array{T,2}) where {T}
    out = zeros(T, size(LE, 2), size(RE, 2), size(B, 3))
    for ol ∈ axes(LE, 1), nl ∈ axes(LE, 2), or ∈ axes(RE, 1), nr ∈ axes(RE, 2), nc ∈ axes(B, 3)
        out[nl, nr, nc] += LE[ol, nl] * B[ol, or, nc] * RE[or, nr]
    end
    out
end

function ref_project_ket_on_bra(LE::Array{T,2}, RE::Array{T,3}) where {T}
    out = zeros(T, size(LE, 2), size(RE, 2), size(RE, 3))
    for ol ∈ axes(LE, 1), nl ∈ axes(LE, 2), nr ∈ axes(RE, 2), nc ∈ axes(RE, 3)
        out[nl, nr, nc] += LE[ol, nl] * RE[ol, nr, nc]
    end
    out
end

function ref_update_reduced_env_right(K::Array{T,1}, RE::Array{T,2}, M::Array{T,4}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 1), size(M, 1))
    for d ∈ axes(M, 2), y ∈ axes(M, 1), β ∈ axes(M, 3), γ ∈ axes(M, 4),
        x ∈ axes(B, 1), α ∈ axes(B, 2)

        out[x, y] += K[d] * M[y, d, β, γ] * B[x, α, γ] * RE[α, β]
    end
    out
end

function ref_update_reduced_env_right(RR::Array{T,2}, M0::Array{T,2}) where {T}
    out = zeros(T, size(RR, 1), size(M0, 1))
    for x ∈ axes(RR, 1), z ∈ axes(RR, 2), y ∈ axes(M0, 1)
        out[x, y] += M0[y, z] * RR[x, z]
    end
    out
end

function ref_contract_tensors43(B::Array{T,4}, A::Array{T,3}) where {T}
    X, Bb = size(A, 1), size(A, 2)
    Y, Z, Aa = size(B, 1), size(B, 2), size(B, 3)
    out = zeros(T, X * Y, Bb * Aa, Z)
    for σ ∈ axes(B, 4), a ∈ 1:Aa, b ∈ 1:Bb, z ∈ 1:Z, y ∈ 1:Y, x ∈ 1:X
        out[x+(y-1)*X, b+(a-1)*Bb, z] += B[y, z, a, σ] * A[x, b, σ]
    end
    out
end

function ref_corner_matrix(C::Array{T,3}, M::Array{T,4}, B::Array{T,3}) where {T}
    out = zeros(T, size(B, 1), size(M, 1), size(C, 2), size(M, 2))
    for ml ∈ axes(M, 1), mt ∈ axes(M, 2), mr ∈ axes(M, 3), mb ∈ axes(M, 4),
        ll ∈ axes(B, 1), rr ∈ axes(B, 2), tt ∈ axes(C, 2)

        out[ll, ml, tt, mt] += M[ml, mt, mr, mb] * B[ll, rr, mb] * C[rr, tt, mr]
    end
    out
end

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

Random.seed!(1234)

@testset verbose = true "dense contraction kernels [$T, $dev]" for T ∈ dense_test_types(),
    dev ∈ dense_test_devices()

    rt = dense_test_rtol(T)
    on(x) = dense_on(x, dev)

    @testset "contract_tensor3_matrix" begin
        A = randn(T, 2, 3, 4)
        M = randn(T, 4, 5)
        ref = ref_contract_tensor3_matrix(A, M)
        out = contract_tensor3_matrix(on(A), on(M))
        @test size(out) == (2, 3, 5)
        @test Array(out) ≈ ref rtol = rt

        # Diagonal matrix variant (MatrixOrCuMatrix includes Diagonal)
        D = Diagonal(randn(T, 4))
        refD = ref_contract_tensor3_matrix(A, Matrix(D))
        outD = contract_tensor3_matrix(on(A), on(D))
        @test size(outD) == (2, 3, 4)
        @test Array(outD) ≈ refD rtol = rt
    end

    @testset "contract_matrix_tensor3" begin
        A = randn(T, 2, 3, 4)
        M = randn(T, 5, 4)          # out leg first, contracted leg second (A * M')
        ref = ref_contract_matrix_tensor3(M, A)
        out = contract_matrix_tensor3(on(M), on(A))
        @test size(out) == (2, 3, 5)
        @test Array(out) ≈ ref rtol = rt

        D = Diagonal(randn(T, 4))
        refD = ref_contract_matrix_tensor3(Matrix(D), A)
        outD = contract_matrix_tensor3(on(D), on(A))
        @test size(outD) == (2, 3, 4)
        @test Array(outD) ≈ refD rtol = rt
    end

    @testset "update_env_left(LE, A, M, B)" begin
        LE = randn(T, 2, 3, 4)      # (ob, ot, oc)
        A = randn(T, 3, 5, 2)       # (ot, nt, α)
        M = randn(T, 4, 2, 5, 3)    # (oc, α, nc, β)
        B = randn(T, 2, 3, 3)       # (ob, nb, β)
        ref = ref_update_env_left(LE, A, M, B)
        out = update_env_left(on(LE), on(A), on(M), on(B))
        @test size(out) == (3, 5, 5) # (nb, nt, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_env_left(LE, A, B)" begin
        LE = randn(T, 2, 3)         # (ob, ot)
        A = randn(T, 3, 4, 2)       # (ot, nt, α)
        B = randn(T, 2, 5, 2)       # (ob, nb, α)
        ref = ref_update_env_left(LE, A, B)
        out = update_env_left(on(LE), on(A), on(B))
        @test size(out) == (5, 4)   # (nb, nt)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_env_left(LE, A) boundary" begin
        LE = randn(T, 4, 2)         # (nb, ot)
        A = randn(T, 2, 3, 5)       # (ot, nt, nc)
        ref = ref_update_env_left(LE, A)
        out = update_env_left(on(LE), on(A))
        @test size(out) == (4, 3, 5) # (nb, nt, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_env_right(RE, A, M, B)" begin
        RE = randn(T, 2, 3, 4)      # (ob, ot, oc)
        A = randn(T, 5, 3, 2)       # (nt, ot, α)
        M = randn(T, 5, 2, 4, 3)    # (nc, α, oc, β)
        B = randn(T, 3, 2, 3)       # (nb, ob, β)
        ref = ref_update_env_right(RE, A, M, B)
        out = update_env_right(on(RE), on(A), on(M), on(B))
        @test size(out) == (3, 5, 5) # (nb, nt, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_env_right(RE, A, B)" begin
        RE = randn(T, 2, 3)         # (ob, ot)
        A = randn(T, 4, 3, 2)       # (nt, ot, α)
        B = randn(T, 5, 2, 2)       # (nb, ob, α)
        ref = ref_update_env_right(RE, A, B)
        out = update_env_right(on(RE), on(A), on(B))
        @test size(out) == (5, 4)   # (nb, nt)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_env_right(RE, C) boundary" begin
        RE = randn(T, 4, 2, 3)      # (nb, ot, oc)
        C = randn(T, 5, 2, 3)       # (nt, ot, oc)
        ref = ref_update_env_right(RE, C)
        out = update_env_right(on(RE), on(C))
        @test size(out) == (4, 5)   # (nb, nt)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "project_ket_on_bra(LE, B, M, RE)" begin
        LE = randn(T, 2, 3, 4)      # (ol, nl, lc)
        B = randn(T, 2, 5, 3)       # (ol, or, oc)
        M = randn(T, 4, 2, 5, 3)    # (lc, nc, rc, oc)
        RE = randn(T, 5, 4, 5)      # (or, nr, rc)
        ref = ref_project_ket_on_bra(LE, B, M, RE)
        out = project_ket_on_bra(on(LE), on(B), on(M), on(RE))
        @test size(out) == (3, 4, 2) # (nl, nr, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "project_ket_on_bra(LE, B, RE)" begin
        LE = randn(T, 2, 4)         # (ol, nl)
        B = randn(T, 2, 3, 2)       # (ol, or, nc)
        RE = randn(T, 3, 5)         # (or, nr)
        ref = ref_project_ket_on_bra(LE, B, RE)
        out = project_ket_on_bra(on(LE), on(B), on(RE))
        @test size(out) == (4, 5, 2) # (nl, nr, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "project_ket_on_bra(LE, RE) boundary" begin
        LE = randn(T, 2, 4)         # (ol, nl)
        RE = randn(T, 2, 3, 2)      # (ol, nr, nc)
        ref = ref_project_ket_on_bra(LE, RE)
        out = project_ket_on_bra(on(LE), on(RE))
        @test size(out) == (4, 3, 2) # (nl, nr, nc)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_reduced_env_right(K, RE, M, B) core" begin
        K = randn(T, 2)             # (d)
        M = randn(T, 3, 2, 4, 2)    # (y, d, β, γ)
        B = randn(T, 5, 3, 2)       # (x, α, γ)
        RE = randn(T, 3, 4)         # (α, β)
        ref = ref_update_reduced_env_right(K, RE, M, B)
        out = update_reduced_env_right(on(K), on(RE), on(M), on(B))
        @test size(out) == (5, 3)   # (x, y)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "update_reduced_env_right(RE, m, MpoTensor, B): ctr only" begin
        ctr = randn(T, 3, 2, 4, 2)  # (y, d, β, γ)
        B = randn(T, 5, 3, 2)       # (x, α, γ)
        RE = randn(T, 3, 4)         # (α, β)
        for m ∈ 1:2
            K = zeros(T, 2)
            K[m] = one(T)
            ref = ref_update_reduced_env_right(K, RE, ctr, B)
            mpo = MpoTensor(TensorMap{T}(0 => copy(ctr)))
            dev == :GPU && move_to_CUDA!(mpo)
            out = update_reduced_env_right(on(RE), m, mpo, on(B))
            @test size(out) == (5, 3)
            @test Array(out) ≈ ref rtol = rt
        end
    end

    @testset "update_reduced_env_right(RE, m, MpoTensor, B): top and bot legs" begin
        t1 = randn(T, 3, 2)         # top[1]; size(Mpo, 2) == 3, so m ∈ 1:3
        t2 = randn(T, 2, 4)         # top[2]; chain output dim 4 == size(ctr, 2)
        ctr = randn(T, 3, 4, 2, 3)  # (y, d, β, γ)
        b1 = randn(T, 3, 2)         # bot[1]; maps ctr leg 4 (dim 3) <- mps phys leg (dim 2)
        B = randn(T, 4, 2, 2)       # (x, α, σ) with σ == size(b1, 2)
        RE = randn(T, 2, 2)         # (α, β)

        # reference, assembled from the defining equations:
        # K[d] = (t1 * t2)[m, d]; B3[x, α, γ] = Σ_σ b1[γ, σ] B[x, α, σ]
        topchain = t1 * t2
        Bd = ref_contract_matrix_tensor3(b1, B)
        for m ∈ 1:3
            K = topchain[m, :]
            ref = ref_update_reduced_env_right(K, RE, ctr, Bd)
            mpo = MpoTensor(
                TensorMap{T}(-2 => copy(t1), -1 => copy(t2), 0 => copy(ctr), 1 => copy(b1)),
            )
            dev == :GPU && move_to_CUDA!(mpo)
            out = update_reduced_env_right(on(RE), m, mpo, on(B))
            @test size(out) == (4, 3)   # (x, y)
            @test Array(out) ≈ ref rtol = rt
        end
    end

    @testset "update_reduced_env_right(RR, M0)" begin
        RR = randn(T, 3, 4)         # (x, z)
        M0 = randn(T, 5, 4)         # (y, z)
        ref = ref_update_reduced_env_right(RR, M0)
        out = update_reduced_env_right(on(RR), on(M0))
        @test size(out) == (3, 5)   # (x, y)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "contract_tensors43" begin
        B = randn(T, 2, 4, 3, 2)    # (y, z, a, σ)
        A = randn(T, 3, 5, 2)       # (x, b, σ)
        ref = ref_contract_tensors43(B, A)
        out = contract_tensors43(on(B), on(A))
        @test size(out) == (6, 15, 4) # ((x,y), (b,a), z)
        @test Array(out) ≈ ref rtol = rt
    end

    @testset "corner_matrix" begin
        C = randn(T, 2, 3, 4)       # (rr, tt, mr)
        M = randn(T, 5, 2, 4, 3)    # (ml, mt, mr, mb)
        B = randn(T, 4, 2, 3)       # (ll, rr, mb)
        ref = ref_corner_matrix(C, M, B)
        out = corner_matrix(on(C), on(M), on(B))
        @test size(out) == (4, 5, 3, 2) # (ll, ml, tt, mt)
        @test Array(out) ≈ ref rtol = rt
    end
end

if !CUDA.functional()
    @info "CUDA not functional -- GPU variants of dense contraction kernels were NOT tested."
end
