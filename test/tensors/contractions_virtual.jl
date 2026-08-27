# Tests for src/contractions/virtual.jl (VirtualTensor kernels):
#   update_env_left, update_env_right, project_ket_on_bra,
#   update_reduced_env_right, contract_tensors43, corner_matrix.
#
# Reference convention (derived from how the kernels consume M.projs / M.con and
# from the Dense-mode materialization in
# src/engine/king_single_node.jl, Val{:virtual_single_node}):
#
#   A VirtualTensor M with projectors (p_lb, p_l, p_lt, p_rb, p_r, p_rt) and
#   central matrix `con` (con[lc, rc]; for a CentralTensor con,
#   con[(l1,l2),(r1,r2)] = e11[l1,r1]*e21[l2,r1]*e12[l1,r2]*e22[l2,r2],
#   with l1 / r1 fastest) represents the dense 4-leg MPO tensor
#
#       M4[l, (lt,rt), r, (lb,rb)] = con[p_l[l], p_r[r]] * δ(lt, p_lt[l]) *
#               δ(rt, p_rt[r]) * δ(lb, p_lb[l]) * δ(rb, p_rb[r])
#
#   with index order [left-central, top, right-central, bottom]. The fused top
#   (lt, rt) and bottom (lb, rb) legs have the LEFT factor fastest
#   (column-major), matching how the kernels reshape the MPS tensors at entry:
#   A :: (Dlt, Drt, slt*srt) -> reshape -> (Dlt, Drt, slt, srt); B likewise.
#
#   Projector invariants (as produced by fuse_projectors / rank_reveal in
#   SpinGlassEngine): all six projectors take values in 1:size(lp, k) and
#   attain the maximum; the three left (right) projectors have equal length
#   nl (nr); and the left (right) triples (p_lb[l], p_l[l], p_lt[l]) are
#   pairwise DISTINCT. The kernels' scatter operations
#   (e.g. tmp1[:, pl_b_ct] = LE[:, ilt, :]) silently overwrite entries of
#   duplicated triples, so distinctness is a hard requirement.
#
#   Environments follow the dense-kernel conventions of
#   src/contractions/dense.jl: LE[b, t, c], RE[b, t, c], and e.g.
#   L[nb, nt, nc] = Σ LE[ob, ot, oc] A[ot, nt, α] M4[oc, α, nc, β] B[ob, nb, β].
#
#   All references below are computed in Float64 with plain @tensor
#   (TensorOperations) on the materialized dense M4 - fully independent of the
#   kernels under test.

module ContractionsVirtualTest

using Test
using TensorOperations
using LinearAlgebra
using Random
using CUDA
using SpinGlassPEPS.SpinGlassTensors

import SpinGlassPEPS.SpinGlassTensors:
    update_env_left,
    update_env_right,
    project_ket_on_bra,
    update_reduced_env_right,
    contract_tensors43,
    corner_matrix,
    CentralTensor,
    VirtualTensor,
    PoolOfProjectors,
    get_projector!,
    move_to_CUDA!

# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

# n pairwise-distinct triples with values (1:s1, 1:s2, 1:s3), each range
# attained (so that size(lp, k) == maximum(p) matches the declared sizes).
function distinct_triples(rng, n, s1, s2, s3)
    space = vec([(a, b, c) for a = 1:s1, b = 1:s2, c = 1:s3])
    @assert max(s1, s2, s3) <= n <= length(space)
    for _ = 1:10_000
        sel = shuffle(rng, space)[1:n]
        if maximum(t -> t[1], sel) == s1 &&
           maximum(t -> t[2], sel) == s2 &&
           maximum(t -> t[3], sel) == s3
            return [t[1] for t in sel], [t[2] for t in sel], [t[3] for t in sel]
        end
    end
    error("could not sample covering distinct projector triples")
end

rand_con(rng, ::Type{T}, slc::Int, src::Int) where {T} = rand(rng, T, slc, src) .- T(0.5)

function rand_central(rng, ::Type{T}, dl::NTuple{2,Int}, dr::NTuple{2,Int}) where {T}
    l1, l2 = dl
    r1, r2 = dr
    CentralTensor(
        rand(rng, T, l1, r1) .- T(0.5),
        rand(rng, T, l1, r2) .- T(0.5),
        rand(rng, T, l2, r1) .- T(0.5),
        rand(rng, T, l2, r2) .- T(0.5),
    )
end

# Fixture: valid VirtualTensor + compatible MPS/environment tensors.
#   s = (slb, slc, slt, srb, src, srt)  -- projector ranges (fused leg sizes)
#   D = (Dlb, Drb, Dlt, Drt)            -- MPS bond dimensions
#   nl, nr                              -- # distinct left / right triples
function make_fixture(::Type{T}, rng; nl, nr, s, D, central = false, srcc = 3) where {T}
    slb, slc, slt, srb, src, srt = s
    Dlb, Drb, Dlt, Drt = D
    lp = PoolOfProjectors{Int}()
    p_lb, p_lc, p_lt = distinct_triples(rng, nl, slb, slc, slt)
    p_rb, p_rc, p_rt = distinct_triples(rng, nr, srb, src, srt)
    con =
        central ? rand_central(rng, T, (slc ÷ 2, 2), (src ÷ 2, 2)) :
        rand_con(rng, T, slc, src)
    M = VirtualTensor(lp, con, (p_lb, p_lc, p_lt, p_rb, p_rc, p_rt))
    @assert size(M) == (nl, slt * srt, nr, slb * srb)
    (;
        M,
        A = rand(rng, T, Dlt, Drt, slt * srt) .- T(0.5),
        B = rand(rng, T, Dlb, Drb, slb * srb) .- T(0.5),
        LE = rand(rng, T, Dlb, Dlt, nl) .- T(0.5),
        RE = rand(rng, T, Drb, Drt, nr) .- T(0.5),
        K = rand(rng, T, slt * srt) .- T(0.5),
        RR = rand(rng, T, Drb, nr) .- T(0.5),
        C = rand(rng, T, Drb, srcc, nr) .- T(0.5),
    )
end

# ---------------------------------------------------------------------------
# independent dense reference
# ---------------------------------------------------------------------------

dense_con(con::AbstractMatrix) = Float64.(Array(con))
function dense_con(con::CentralTensor)
    e11, e12, e21, e22 = map(x -> Float64.(Array(x)), (con.e11, con.e12, con.e21, con.e22))
    sl1, sr1 = size(e11)
    sl2, sr2 = size(e22)
    E = [
        e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2] for l1 = 1:sl1, l2 = 1:sl2,
        r1 = 1:sr1, r2 = 1:sr2
    ]
    reshape(E, sl1 * sl2, sr1 * sr2)
end

# Materialize the defining dense 4-leg MPO tensor M4[l, t, r, b] (Float64).
function dense_virtual(M::VirtualTensor)
    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt =
        (Array(get_projector!(M.lp, k, :CPU)) for k in M.projs)
    slb, slt = size(M.lp, M.projs[1]), size(M.lp, M.projs[3])
    srb, srt = size(M.lp, M.projs[4]), size(M.lp, M.projs[6])
    con = dense_con(M.con)
    nl, nr = length(p_lc), length(p_rc)
    M4 = zeros(Float64, nl, slt, srt, nr, slb, srb)
    for l = 1:nl, r = 1:nr
        M4[l, p_lt[l], p_rt[r], r, p_lb[l], p_rb[r]] = con[p_lc[l], p_rc[r]]
    end
    reshape(M4, nl, slt * srt, nr, slb * srb)
end

f64(x) = Float64.(Array(x))

function references(fx)
    M4 = dense_virtual(fx.M)
    A, B = f64(fx.A), f64(fx.B)
    LE, RE = f64(fx.LE), f64(fx.RE)
    K, RR, C = f64(fx.K), f64(fx.RR), f64(fx.C)

    @tensor Lref[nb, nt, nc] :=
        LE[ob, ot, oc] * A[ot, nt, α] * M4[oc, α, nc, β] * B[ob, nb, β]
    @tensor Rref[nb, nt, nc] :=
        RE[ob, ot, oc] * A[nt, ot, α] * M4[nc, α, oc, β] * B[nb, ob, β]
    @tensor PKBref[nl, nr, nc] :=
        LE[ol, nl, lc] * B[ol, or, oc] * M4[lc, nc, rc, oc] * RE[or, nr, rc]
    @tensor RRref[x, y] := K[d] * M4[y, d, β, γ] * B[x, α, γ] * RR[α, β]
    @tensor C43ref[x, y, b, a, z] := M4[y, z, a, σ] * B[x, b, σ]
    C43ref =
        reshape(C43ref, size(C43ref, 1) * size(C43ref, 2), size(C43ref, 3) * size(C43ref, 4), :)
    @tensor CMref[ll, ml, tt, mt] := M4[ml, mt, mr, mb] * B[ll, rr, mb] * C[rr, tt, mr]
    (; Lref, Rref, PKBref, RRref, C43ref, CMref)
end

# ---------------------------------------------------------------------------
# kernel-vs-reference checks (CPU or GPU, controlled by `gpu`)
# ---------------------------------------------------------------------------

to_dev(x::AbstractArray, gpu::Bool) = gpu ? CuArray(x) : x

function kernel_checks(fx, ref, rtol; gpu = false)
    M = gpu ? move_to_CUDA!(deepcopy(fx.M)) : fx.M
    A, B = to_dev(fx.A, gpu), to_dev(fx.B, gpu)
    LE, RE = to_dev(fx.LE, gpu), to_dev(fx.RE, gpu)
    K, RR, C = to_dev(fx.K, gpu), to_dev(fx.RR, gpu), to_dev(fx.C, gpu)

    @testset "update_env_left" begin
        L = update_env_left(LE, A, M, B)
        @test size(L) == size(ref.Lref)
        @test isapprox(f64(L), ref.Lref; rtol = rtol)
    end
    @testset "update_env_right" begin
        R = update_env_right(RE, A, M, B)
        @test size(R) == size(ref.Rref)
        @test isapprox(f64(R), ref.Rref; rtol = rtol)
    end
    @testset "project_ket_on_bra" begin
        P = project_ket_on_bra(LE, B, M, RE)
        @test size(P) == size(ref.PKBref)
        @test isapprox(f64(P), ref.PKBref; rtol = rtol)
    end
    @testset "update_reduced_env_right" begin
        RRnew = update_reduced_env_right(K, RR, M, B)
        @test size(RRnew) == size(ref.RRref)
        @test isapprox(f64(RRnew), ref.RRref; rtol = rtol)
    end
    @testset "contract_tensors43" begin
        C43 = contract_tensors43(M, B)
        @test size(C43) == size(ref.C43ref)
        @test isapprox(f64(C43), ref.C43ref; rtol = rtol)
    end
    @testset "corner_matrix" begin
        CM = corner_matrix(C, M, B)
        @test size(CM) == size(ref.CMref)
        @test isapprox(f64(CM), ref.CMref; rtol = rtol)
    end
end

# ---------------------------------------------------------------------------
# fixtures
#
# Branch coverage of the size-based dispatch inside the kernels
# (slpb = size(lp, p_lb) etc.):
#   update_env_left          : slpb * srpt >= slpt * srpb  -> fixture A / else B
#   project_ket_on_bra       : slpb >= srpb                -> fixture A / else B
#   update_env_right         : srpb * slpt >= srpt * slpb  -> fixture B / else A
#   update_reduced_env_right : srpb * slpt >= srpt * slpb  -> fixture B / else A
#   contract_tensors43       : size(con,1) <= size(con,2)  -> fixture B / else A
# Fixture C uses a CentralTensor as M.con (equal sizes: all ">=" branches) and
# additionally exercises batched_mul!(:, :, ::CentralTensor) and its adjoint.
# Fixture D has trivial (singleton) legs: slc = srt = 1 and a bond dim of 1.
# ---------------------------------------------------------------------------

const FIXTURES = (
    #        (slb, slc, slt, srb, src, srt)   (Dlb, Drb, Dlt, Drt)  nl  nr central
    ("A", (3, 3, 2, 2, 2, 3), (2, 3, 3, 2), 6, 5, false),
    ("B", (2, 2, 3, 3, 4, 2), (3, 2, 2, 3), 6, 5, false),
    ("C-central", (2, 4, 2, 2, 2, 2), (3, 2, 2, 3), 6, 5, true),
    ("D-degenerate", (2, 1, 2, 2, 2, 1), (1, 2, 2, 1), 3, 2, false),
)

@testset "VirtualTensor contractions vs dense reference" begin
    for (name, s, D, nl, nr, central) in FIXTURES,
        T in (Float64, Float32)

        rng = MersenneTwister(hash((name, T === Float64)) % 10_000 + 7)
        fx = make_fixture(T, rng; nl = nl, nr = nr, s = s, D = D, central = central)
        ref = references(fx)
        rtol = T === Float64 ? 1e-12 : 1e-4

        @testset "$name $T CPU" begin
            kernel_checks(fx, ref, rtol; gpu = false)
        end

        if CUDA.functional()
            @testset "$name $T GPU" begin
                kernel_checks(fx, ref, rtol; gpu = true)
            end
        end
    end

    if !CUDA.functional()
        @warn "CUDA not functional - GPU variants of the VirtualTensor kernels were NOT tested"
    end
end

end # module
