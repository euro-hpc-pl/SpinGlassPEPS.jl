# contractions_site.jl
#
# Unit tests for the SiteTensor contraction kernels in src/contractions/site.jl:
#   update_env_left, update_env_right, project_ket_on_bra,
#   update_reduced_env_right (incl. the MpoTensor wrapper), contract_tensors43,
#   corner_matrix.
#
# Reference convention (derived from src/base.jl, the dense kernels in
# src/contractions/dense.jl, and the Dense-sparsity materialization in
# src/engine/tensors.jl, `Val{:site}`):
#
#   SiteTensor(lp, loc_exp, (pl, pt, pr, pb)) represents the dense 4-leg tensor
#       Md[l, t, r, b] = Σ_k loc_exp[k] δ(l, pl[k]) δ(t, pt[k]) δ(r, pr[k]) δ(b, pb[k])
#   with leg order (left, top, right, bottom); size(Md, i) == maximum(projs[i])
#   and repeated projector columns ACCUMULATE (+=).
#
# Expected values are computed with plain @tensor (TensorOperations) on the
# materialized dense tensor, using the defining equations of the dense family:
#   update_env_left:   L[nb,nt,nc] = LE[ob,ot,oc] A[ot,nt,a] Md[oc,a,nc,b] B[ob,nb,b]
#   update_env_right:  R[nb,nt,nc] = RE[ob,ot,oc] A[nt,ot,a] Md[nc,a,oc,b] B[nb,ob,b]
#   project_ket_on_bra: O[nl,nr,nc] = LE[ol,nl,lc] B[ol,or,oc] Md[lc,nc,rc,oc] RE[or,nr,rc]
#   update_reduced_env_right: R[x,y] = K[d] Md[y,d,b,g] B[x,a,g] RE[a,b]
#   contract_tensors43: C[(x,l),(b,r),t] = Σ_σ Md[l,t,r,σ] B[x,b,σ]  (column-major merge)
#   corner_matrix:     Cn[ll,ml,tt,mt] = Md[ml,mt,mr,mb] B[ll,rr,mb] C[rr,tt,mr]
#
# The reference never touches contract_sparse_with_three nor the memoized
# SparseArrays.sparse projector matrices. As an extra cross-check, each kernel
# is also compared with the dense-family method dispatched on Md::Array{T,4}.
#
# GPU variants are run (behind CUDA.functional()) on the same fixtures moved
# with the package's own move_to_CUDA! / CuArray, and compared to the same
# CPU-dense references.

using Test
using SpinGlassPEPS.SpinGlassTensors
using TensorOperations
using LinearAlgebra
using CUDA
using Random

using SpinGlassPEPS.SpinGlassTensors:
    update_env_left,
    update_env_right,
    project_ket_on_bra,
    update_reduced_env_right,
    contract_tensors43,
    corner_matrix

"""
Build a SiteTensor with valid projector structure: 4 projector vectors of a
common length K, values in 1:D_i, maximum D_i attained (so dims == (Dl,Dt,Dr,Db)),
and one deliberately duplicated projector column (k=2 and k=3) to exercise
accumulation in the scatter step.
"""
function make_site_fixture(::Type{T}, rng; Dl, Dt, Dr, Db, K) where {T<:Real}
    @assert K >= 3
    pl = rand(rng, 1:Dl, K)
    pt = rand(rng, 1:Dt, K)
    pr = rand(rng, 1:Dr, K)
    pb = rand(rng, 1:Db, K)
    pl[1], pt[1], pr[1], pb[1] = Dl, Dt, Dr, Db  # guarantee maxima are attained
    pl[3], pt[3], pr[3], pb[3] = pl[2], pt[2], pr[2], pb[2]  # duplicated column
    loc_exp = rand(rng, T, K) .+ T(1 // 10)  # positive, like exp(-β E) in production
    lp = PoolOfProjectors{Int}()
    SiteTensor(lp, loc_exp, (pl, pt, pr, pb))
end

"""
Materialize the dense 4-leg tensor represented by a SiteTensor, independently
of the contraction kernels: Md[pl[k], pt[k], pr[k], pb[k]] += loc_exp[k].
"""
function dense_site_tensor(M::SiteTensor{T,4}) where {T<:Real}
    ps = map(k -> Array(get_projector!(M.lp, k, :CPU)), M.projs)
    le = Array(M.loc_exp)
    Md = zeros(T, size(M))
    for k in eachindex(le)
        @inbounds Md[ps[1][k], ps[2][k], ps[3][k], ps[4][k]] += le[k]
    end
    Md
end

function test_site_kernels(::Type{T}; Dl, Dt, Dr, Db, K, seed) where {T<:Real}
    rng = MersenneTwister(seed)
    rtol = T === Float64 ? 1e-10 : 1e-4

    M = make_site_fixture(T, rng; Dl, Dt, Dr, Db, K)
    Md = dense_site_tensor(M)

    @testset "fixture invariants" begin
        ps = map(k -> Array(get_projector!(M.lp, k, :CPU)), M.projs)
        @test all(length.(ps) .== length(M.loc_exp))
        for (i, p) in enumerate(ps)
            @test all(1 .<= p .<= size(M.lp, M.projs[i]))
            @test maximum(p) == size(M, i)
        end
        @test size(M) == (Dl, Dt, Dr, Db)
        @test size(Md) == size(M)
        @test sum(Md) ≈ sum(Array(M.loc_exp)) rtol = rtol  # accumulation preserved
    end

    # pairwise-distinct bond dims to catch any leg-order mistake
    bBo, bAo, bAn, bBn = 2, 3, 4, 5

    # ---- inputs and independent dense references ------------------------------
    # update_env_left
    LE_l = rand(rng, T, bBo, bAo, Dl)
    A_l = rand(rng, T, bAo, bAn, Dt)
    B_l = rand(rng, T, bBo, bBn, Db)
    @tensor ref_l[nb, nt, nc] :=
        LE_l[ob, ot, oc] * A_l[ot, nt, α] * Md[oc, α, nc, β] * B_l[ob, nb, β]

    # update_env_right
    RE_r = rand(rng, T, bBo, bAo, Dr)
    A_r = rand(rng, T, bAn, bAo, Dt)
    B_r = rand(rng, T, bBn, bBo, Db)
    @tensor ref_r[nb, nt, nc] :=
        RE_r[ob, ot, oc] * A_r[nt, ot, α] * Md[nc, α, oc, β] * B_r[nb, ob, β]

    # project_ket_on_bra
    LE_p = rand(rng, T, bBo, bAo, Dl)
    B_p = rand(rng, T, bBo, bBn, Db)
    RE_p = rand(rng, T, bBn, bAn, Dr)
    @tensor ref_p[nl, nr, nc] :=
        LE_p[ol, nl, lc] * B_p[ol, or, oc] * Md[lc, nc, rc, oc] * RE_p[or, nr, rc]

    # update_reduced_env_right
    K_u = rand(rng, T, Dt)
    RE_u = rand(rng, T, bBo, Dr)
    B_u = rand(rng, T, bBn, bBo, Db)
    @tensor ref_u[x, y] := K_u[d] * Md[y, d, β, γ] * B_u[x, α, γ] * RE_u[α, β]

    # contract_tensors43
    sb1, sb2 = 2, 3
    B_c = rand(rng, T, sb1, sb2, Db)
    @tensor ref_c5[x, l, b, r, t] := Md[l, t, r, σ] * B_c[x, b, σ]
    ref_c = reshape(ref_c5, sb1 * Dl, sb2 * Dr, Dt)

    # corner_matrix
    ll, rr_, tt = 2, 3, 4
    B_m = rand(rng, T, ll, rr_, Db)
    C_m = rand(rng, T, rr_, tt, Dr)
    @tensor ref_m[l1, ml, t1, mt] :=
        Md[ml, mt, mr, mb] * B_m[l1, r1, mb] * C_m[r1, t1, mr]

    # ---- CPU kernels ----------------------------------------------------------
    @testset "CPU update_env_left" begin
        res = update_env_left(LE_l, A_l, M, B_l)
        @test res isa Array{T,3}
        @test size(res) == (bBn, bAn, Dr)
        @test isapprox(res, ref_l; rtol)
        @test isapprox(update_env_left(LE_l, A_l, Md, B_l), ref_l; rtol) # dense-family cross-check
    end

    @testset "CPU update_env_right" begin
        res = update_env_right(RE_r, A_r, M, B_r)
        @test res isa Array{T,3}
        @test size(res) == (bBn, bAn, Dl)
        @test isapprox(res, ref_r; rtol)
        @test isapprox(update_env_right(RE_r, A_r, Md, B_r), ref_r; rtol)
    end

    @testset "CPU project_ket_on_bra" begin
        res = project_ket_on_bra(LE_p, B_p, M, RE_p)
        @test res isa Array{T,3}
        @test size(res) == (bAo, bAn, Dt)
        @test isapprox(res, ref_p; rtol)
        @test isapprox(project_ket_on_bra(LE_p, B_p, Md, RE_p), ref_p; rtol)
    end

    @testset "CPU update_reduced_env_right" begin
        res = update_reduced_env_right(K_u, RE_u, M, B_u)
        @test res isa Array{T,2}
        @test size(res) == (bBn, Dl)
        @test isapprox(res, ref_u; rtol)
        @test isapprox(update_reduced_env_right(K_u, RE_u, Md, B_u), ref_u; rtol)
    end

    @testset "CPU update_reduced_env_right (MpoTensor wrapper)" begin
        mt = MpoTensor(TensorMap{T}(0 => M))
        @test size(mt) == size(M)
        for m in (1, Dt)
            Kd = zeros(T, Dt)
            Kd[m] = one(T)
            @tensor ref_w[x, y] := Kd[d] * Md[y, d, β, γ] * B_u[x, α, γ] * RE_u[α, β]
            res = update_reduced_env_right(RE_u, m, mt, B_u)
            @test size(res) == (bBn, Dl)
            @test isapprox(res, ref_w; rtol)
        end
    end

    @testset "CPU contract_tensors43" begin
        res = contract_tensors43(M, B_c)
        @test res isa Array{T,3}
        @test size(res) == (sb1 * Dl, sb2 * Dr, Dt)
        @test isapprox(res, ref_c; rtol)
        @test isapprox(contract_tensors43(Md, B_c), ref_c; rtol)
    end

    @testset "CPU corner_matrix" begin
        # dense-family kernel on the materialized tensor matches the reference
        @test isapprox(corner_matrix(C_m, Md, B_m), ref_m; rtol)
        # KNOWN DEFECT (as of writing): the SiteTensor method computes
        # `ip12 * outp'` (src/contractions/site.jl:193) where `outp` is still
        # the 3-D batched array -- the flattened 2-D matrix was assigned to
        # `Bp` on line 189 -- so `adjoint` throws for every input. Once fixed
        # (`ip12 * Bp'`), the result must equal ref_m below.
        res = nothing
        threw = false
        try
            res = corner_matrix(C_m, M, B_m)
        catch
            threw = true
        end
        if threw
            @test_broken !threw
        else
            @test size(res) == (ll, Dl, tt, Dt)
            @test isapprox(Array(res), ref_m; rtol)
        end
    end

    # ---- GPU kernels: same fixtures moved with the package's own tools --------
    if CUDA.functional()
        move_to_CUDA!(M)
        @test M.loc_exp isa CuArray{T,1}
        @test which_device(M) == Set((:GPU,))

        @testset "GPU update_env_left" begin
            res = update_env_left(CuArray(LE_l), CuArray(A_l), M, CuArray(B_l))
            @test res isa CuArray
            @test size(res) == (bBn, bAn, Dr)
            @test isapprox(Array(res), ref_l; rtol)
        end

        @testset "GPU update_env_right" begin
            res = update_env_right(CuArray(RE_r), CuArray(A_r), M, CuArray(B_r))
            @test res isa CuArray
            @test size(res) == (bBn, bAn, Dl)
            @test isapprox(Array(res), ref_r; rtol)
        end

        @testset "GPU project_ket_on_bra" begin
            res = project_ket_on_bra(CuArray(LE_p), CuArray(B_p), M, CuArray(RE_p))
            @test res isa CuArray
            @test size(res) == (bAo, bAn, Dt)
            @test isapprox(Array(res), ref_p; rtol)
        end

        @testset "GPU update_reduced_env_right" begin
            res = update_reduced_env_right(CuArray(K_u), CuArray(RE_u), M, CuArray(B_u))
            @test res isa CuArray
            @test size(res) == (bBn, Dl)
            @test isapprox(Array(res), ref_u; rtol)
        end

        @testset "GPU update_reduced_env_right (MpoTensor wrapper)" begin
            mt = MpoTensor(TensorMap{T}(0 => M))
            m = Dt
            Kd = zeros(T, Dt)
            Kd[m] = one(T)
            @tensor ref_w[x, y] := Kd[d] * Md[y, d, β, γ] * B_u[x, α, γ] * RE_u[α, β]
            res = update_reduced_env_right(CuArray(RE_u), m, mt, CuArray(B_u))
            @test size(res) == (bBn, Dl)
            @test isapprox(Array(res), ref_w; rtol)
        end

        @testset "GPU contract_tensors43" begin
            res = contract_tensors43(M, CuArray(B_c))
            @test res isa CuArray
            @test size(res) == (sb1 * Dl, sb2 * Dr, Dt)
            @test isapprox(Array(res), ref_c; rtol)
        end

        @testset "GPU corner_matrix" begin
            # same defect as on CPU: `outp'` on a 3-D CuArray throws
            res = nothing
            threw = false
            try
                res = corner_matrix(CuArray(C_m), M, CuArray(B_m))
            catch
                threw = true
            end
            if threw
                @test_broken !threw
            else
                @test size(res) == (ll, Dl, tt, Dt)
                @test isapprox(Array(res), ref_m; rtol)
            end
        end
    end
    nothing
end

@testset "SiteTensor contraction kernels vs independent dense reference ($T)" for T in (
    Float64,
    Float32,
)
    # two shape configurations, all mode sizes pairwise distinct where possible
    test_site_kernels(T; Dl = 3, Dt = 2, Dr = 4, Db = 3, K = 10, seed = 1234)
    test_site_kernels(T; Dl = 2, Dt = 3, Dr = 2, Db = 5, K = 14, seed = 4321)
end
