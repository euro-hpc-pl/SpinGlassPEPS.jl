# virtual.jl: contractions with VirtualTensor on CPU and CUDA
# export update_env_left2, update_env_right2, project_ket_on_bra2

# @memoize Dict
alloc_undef(R, onGPU, shape) = onGPU ? CuArray{R}(undef, shape) : Array{R}(undef, shape)
alloc_zeros(R, onGPU, shape) = onGPU ? CUDA.zeros(R, shape) : zeros(R, shape)

function proj_out(lp, k1, k2, k3, device)
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, device)
    p3 = get_projector!(lp, k3, device)
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = size(lp, k1), size(lp, k2)
    p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
end

function proj_2step_12(lp, (k1, k2), k3, device)
    p1 = get_projector!(lp, k1, :CPU)
    p2 = get_projector!(lp, k2, :CPU)
    p3 = get_projector!(lp, k3, device)
    @assert length(p1) == length(p2) == length(p3)
    s1 = size(lp, k1)

    p12, transitions_matrix = rank_reveal(hcat(p1, p2), :PE)
    (p1, p2) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))

    s12 = maximum(p12)

    if device == :CPU
        p12 = CuArray(p12)
        p1 = CuArray(p1)
        p2 = CuArray(p2)
    end

    pf1 = p12 .+ s12 .* (p3 .- 1)
    pf2 = p1 .+ s1 .* (p2 .- 1)

    pf1, pf2, s12
end

function proj_2step_23(lp, k1, (k2, k3), device)
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, :CPU)
    p3 = get_projector!(lp, k3, :CPU)
    @assert length(p1) == length(p2) == length(p3)

    s1, s2 = size(lp, k1), size(lp, k2)

    p23, transitions_matrix = rank_reveal(hcat(p2, p3), :PE)
    (p2, p3) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))

    s23 = maximum(p23)

    if device == :CPU
        p23 = CuArray(p23)
        p2 = CuArray(p2)
        p3 = CuArray(p3)
    end

    pf1 = p1 .+ s1 .* (p23 .- 1)
    pf2 = p2 .+ s2 .* (p3 .- 1)

    pf1, pf2, s23
end

function merge_projectors_inter(lp, p1, p2, p3, onGPU; order = "1_23")
    s1 = size(lp, p1)
    s2 = size(lp, p2)
    device = onGPU ? :GPU : :CPU
    p1 = get_projector!(lp, p1, device)
    p2 = get_projector!(lp, p2, :CPU)
    p3 = get_projector!(lp, p3, :CPU)

    p23, transitions_matrix = rank_reveal(hcat(p2, p3), :PE)
    s23 = maximum(p23)
    (p2, p3) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    if onGPU
        p23 = CuArray(p23)
        p2 = CuArray(p2)
        p3 = CuArray(p3)
    end
    p2_3 = p2 .+ s2 .* (p3 .- 1)
    p123 = order == "1_23" ? p1 .+ s1 .* (p23 .- 1) : p23 .+ s23 .* (p1 .- 1)  # else "23_1"
    p123, p2_3, s23
end

function update_env_left(
    LE::S,
    A::S,
    M::VirtualTensor{R,4},
    B::S,
) where {S<:Tensor{R,3}} where {R<:Real}
    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt = M.projs
    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    src = length(M.lp, p_rc)

    slpb, slpc, slpt = size(M.lp, p_lb), size(M.lp, p_lc), size(M.lp, p_lt)
    srpb, srpc, srpt = size(M.lp, p_rb), size(M.lp, p_rc), size(M.lp, p_rt)

    onGPU = typeof(LE) <: CuArray

    A = reshape(A, (slt, srt, slpt, srpt))
    B = reshape(B, (slb, srb, slpb, srpb))
    Lout = alloc_zeros(R, onGPU, (srb, srt, src))

    if slpb * srpt >= slpt * srpb
        pl_b_ct, pl_c_t, slpct =
            merge_projectors_inter(M.lp, p_lb, p_lc, p_lt, onGPU; order = "1_23")
        pr_bc_t, pr_b_c, srpbc =
            merge_projectors_inter(M.lp, p_rt, p_rb, p_rc, onGPU; order = "23_1")

        B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lpb, rb, rpb]
        B2 = reshape(B2, (slb * slpb, srb * srpb))  # [(lb, lpb), (rb, rpb)]

        tmp1 = alloc_zeros(R, onGPU, (slb, slpb * slpct))
        tmp2 = alloc_undef(R, onGPU, (srb * srpb, slpct))
        tmp3 = alloc_zeros(R, onGPU, (srb * srpb, slpt * slpc))
        tmp5 = alloc_undef(R, onGPU, (srb * srpb, srpc, slpt))
        tmp8 = alloc_undef(R, onGPU, (srb * srpbc, srpt))

        for ilt ∈ 1:slt
            tmp1[:, pl_b_ct] = LE[:, ilt, :]  # [lb, (lpb, lpct)]
            mul!(tmp2, B2', reshape(tmp1, (slb * slpb, slpct)))  # [(rb, rpb), lpct]
            tmp3[:, pl_c_t] = tmp2  # [(rb, rpb), (lpc, lpt)]
            tmp4 = reshape(tmp3, (srb * srpb, slpc, slpt))  # [(rb, rpb), lpc, lpt]
            batched_mul!(tmp5, tmp4, M.con)
            tmp6 = reshape(tmp5, (srb, srpb * srpc, slpt))  # [rb, (rpb, rpc), lpt]
            tmp7 = reshape(tmp6[:, pr_b_c, :], (srb * srpbc, slpt))  # [(rb, rpbc), lpt]
            for irt ∈ 1:srt
                mul!(tmp8, tmp7, A[ilt, irt, :, :])
                tmp9 = reshape(tmp8, (srb, srpbc * srpt))
                Lout[:, irt, :] .+= tmp9[:, pr_bc_t]  # [rb, rc]
            end
        end
    else
        pl_t_cb, pl_c_b, slpcb =
            merge_projectors_inter(M.lp, p_lt, p_lc, p_lb, onGPU; order = "1_23")
        pr_tc_b, pr_t_c, srptc =
            merge_projectors_inter(M.lp, p_rb, p_rt, p_rc, onGPU; order = "23_1")

        A2 = permutedims(A, (1, 3, 2, 4))  # [lt, lpt, rt, rpt]
        A2 = reshape(A2, (slt * slpt, srt * srpt))  # [(lt, lpt), (rt, rpt)]

        tmp1 = alloc_zeros(R, onGPU, (slt, slpt * slpcb))
        tmp2 = alloc_undef(R, onGPU, (srt * srpt, slpcb))
        tmp3 = alloc_zeros(R, onGPU, (srt * srpt, slpc * slpb))
        tmp5 = alloc_undef(R, onGPU, (srt * srpt, srpc, slpb))
        tmp8 = alloc_zeros(R, onGPU, (srt * srptc, srpb))

        for ilb ∈ 1:slb
            tmp1[:, pl_t_cb] = LE[ilb, :, :]  # [lt, (lpt, lpcb)]
            mul!(tmp2, A2', reshape(tmp1, (slt * slpt, slpcb)))  # [(rt, rpt), lpcb]
            tmp3[:, pl_c_b] = tmp2
            tmp4 = reshape(tmp3, (srt * srpt, slpc, slpb))  # [(rt, rpt), lpc, lpb]
            batched_mul!(tmp5, tmp4, M.con)  # [(rt, rpt), lpb, rpc]
            tmp6 = reshape(tmp5, (srt, srpt * srpc, slpb))  # [(rt, rpt * rpc), lcb]
            tmp7 = reshape(tmp6[:, pr_t_c, :], (srt * srptc, slpb))  # [(rt, rptc), lpb]
            for irb ∈ 1:srb
                mul!(tmp8, tmp7, B[ilb, irb, :, :])
                tmp9 = reshape(tmp8, (srt, srptc * srpb))
                Lout[irb, :, :] .+= tmp9[:, pr_tc_b]  # [rt, rc]
            end
        end
    end
    Lout
end


function project_ket_on_bra(
    LE::S,
    B::S,
    M::VirtualTensor{R,4},
    RE::S,
) where {S<:Tensor{R,3}} where {R<:Real}
    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt = M.projs
    slb, slt = size(LE, 1), size(LE, 2)
    srb, srt = size(RE, 1), size(RE, 2)
    slpb, slpc, slpt = size(M.lp, p_lb), size(M.lp, p_lc), size(M.lp, p_lt)
    srpb, srpc, srpt = size(M.lp, p_rb), size(M.lp, p_rc), size(M.lp, p_rt)

    onGPU = typeof(LE) <: CuArray

    B = reshape(B, (slb, srb, slpb, srpb))
    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lpb, rb, rpb]
    B2 = reshape(B2, (slb * slpb, srb * srpb))  # [(lb, lpb), (rb, rpb)]
    LR = alloc_zeros(R, onGPU, (slt, srt, slpt, srpt))

    if slpb >= srpb
        pl_b_ct, pl_c_t, slpct =
            merge_projectors_inter(M.lp, p_lb, p_lc, p_lt, onGPU; order = "1_23")
        pr_bc_t, pr_b_c, srpbc =
            merge_projectors_inter(M.lp, p_rt, p_rb, p_rc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (slb, slpb * slpct))
        tmp2 = alloc_undef(R, onGPU, (srb * srpb, slpct))
        tmp3 = alloc_zeros(R, onGPU, (srb * srpb, slpc * slpt))
        tmp5 = alloc_undef(R, onGPU, (srb * srpb, srpc, slpt))
        tmp8 = alloc_zeros(R, onGPU, (srb, srpbc * srpt))
        for ilt ∈ 1:slt
            tmp1[:, pl_b_ct] = LE[:, ilt, :]  # [lb, (lpb, lpct)]
            mul!(tmp2, B2', reshape(tmp1, (slb * slpb, slpct)))  # [(rb, rpb), lpct]
            tmp3[:, pl_c_t] = tmp2  # [(rb, rpb), (lpc, lpt)]
            tmp4 = reshape(tmp3, (srb * srpb, slpc, slpt))  # [(rb, rpb), lpc, lpt]
            batched_mul!(tmp5, tmp4, M.con)  # [(rb, rpb), rpc, lpt]
            tmp6 = reshape(tmp5, (srb, srpb * srpc, slpt))  # [rb, (rpb, rpc), lpt]
            tmp7 = reshape(tmp6[:, pr_b_c, :], (srb * srpbc, slpt))  # [(rb, rpbc), lpt]
            for irt ∈ 1:srt
                tmp8[:, pr_bc_t] = RE[:, irt, :]  # [rb, (rpbc, rpt)]
                LR[ilt, irt, :, :] = tmp7' * reshape(tmp8, (srb * srpbc, srpt))  # [lpt, rpt]
            end
        end
    else
        pr_b_ct, pr_c_t, srpct =
            merge_projectors_inter(M.lp, p_rb, p_rc, p_rt, onGPU; order = "1_23")
        pl_bc_t, pl_b_c, slpbc =
            merge_projectors_inter(M.lp, p_lt, p_lb, p_lc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (srb, srpb * srpct))
        tmp2 = alloc_undef(R, onGPU, (slb * slpb, srpct))
        tmp3 = alloc_zeros(R, onGPU, (slb * slpb, srpc * srpt))
        tmp5 = alloc_undef(R, onGPU, (slb * slpb, slpc, srpt))
        tmp8 = alloc_zeros(R, onGPU, (slb, slpbc * slpt))
        for irt ∈ 1:srt
            tmp1[:, pr_b_ct] = RE[:, irt, :]  # [rb, (rpb, rpct)]
            mul!(tmp2, B2, reshape(tmp1, (srb * srpb, srpct)))  # [(lb, lpb), rpct]
            tmp3[:, pr_c_t] = tmp2  # [(lb, lpb), (rpc, rpt)]
            tmp4 = reshape(tmp3, (slb * slpb, srpc, srpt))  # [(lb, lpb), rpc, rpt]
            batched_mul!(tmp5, tmp4, M.con')  # [(lb, lpb), lpc, rpt]
            tmp6 = reshape(tmp5, (slb, slpb * slpc, srpt))  # [lb, (lpb, lpc), rpt]
            tmp7 = reshape(tmp6[:, pl_b_c, :], (slb * slpbc, srpt))  # [(lb, lpbc), rpt]
            for ilt ∈ 1:slt
                tmp8[:, pl_bc_t] = LE[:, ilt, :]  # [lb, (lpbc, lpt)]
                LR[ilt, irt, :, :] = reshape(tmp8, (slb * slpbc, slpt))' * tmp7  # [lct, rct]
            end
        end
    end
    reshape(LR, (slt, srt, slpt * srpt))
end


function update_env_right(
    RE::S,
    A::S,
    M::VirtualTensor{R,4},
    B::S,
) where {S<:Tensor{R,3}} where {R<:Real}
    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slc = length(M.lp, p_lc)

    slpb, slpc, slpt = size(M.lp, p_lb), size(M.lp, p_lc), size(M.lp, p_lt)
    srpb, srpc, srpt = size(M.lp, p_rb), size(M.lp, p_rc), size(M.lp, p_rt)

    onGPU = typeof(RE) <: CuArray

    A = reshape(A, (slt, srt, slpt, srpt))
    B = reshape(B, (slb, srb, slpb, srpb))
    Rout = alloc_zeros(R, onGPU, (slb, slt, slc))

    if srpb * slpt >= srpt * slpb
        B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lpb, rb, rpb]
        B2 = reshape(B2, (slb * slpb, srb * srpb))  # [(lb, lpb), (rb, rpb)]

        pr_b_ct, pr_c_t, srpct =
            merge_projectors_inter(M.lp, p_rb, p_rc, p_rt, onGPU; order = "1_23")
        pl_bc_t, pl_b_c, slpbc =
            merge_projectors_inter(M.lp, p_lt, p_lb, p_lc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (srb, srpb * srpct))
        tmp2 = alloc_undef(R, onGPU, (slb * slpb, srpct))
        tmp3 = alloc_zeros(R, onGPU, (slb * slpb, srpc * srpt))
        tmp5 = alloc_undef(R, onGPU, (slb * slpb, slpc, srpt))
        tmp8 = alloc_undef(R, onGPU, (slb * slpbc, slpt))

        for irt ∈ 1:srt
            tmp1[:, pr_b_ct] = RE[:, irt, :]  # [rb, (rpb, rpct)]
            mul!(tmp2, B2, reshape(tmp1, (srb * srpb, srpct)))  # [(lb, lpb), rpct]
            tmp3[:, pr_c_t] = tmp2  # [(lb, lpb), (rpc, rpt)]
            tmp4 = reshape(tmp3, (slb * slpb, srpc, srpt))  # [(lb, lpb), rpc, rpt]
            batched_mul!(tmp5, tmp4, M.con')
            tmp6 = reshape(tmp5, (slb, slpb * slpc, srpt))  # [lb, (lpb, lpc), rpt]
            tmp7 = reshape(tmp6[:, pl_b_c, :], (slb * slpbc, srpt))  # [(lb, lpbc), rpt]
            for ilt ∈ 1:slt
                mul!(tmp8, tmp7, A[ilt, irt, :, :]')
                tmp9 = reshape(tmp8, (slb, slpbc * slpt))
                Rout[:, ilt, :] .+= tmp9[:, pl_bc_t]
            end
        end
    else
        A2 = permutedims(A, (1, 3, 2, 4))  # [lt, lpt, rt, rpt]
        A2 = reshape(A2, (slt * slpt, srt * srpt))  # [(lt, lpt), (rt, rpt)]

        pr_t_cb, pr_c_b, srpcb =
            merge_projectors_inter(M.lp, p_rt, p_rc, p_rb, onGPU; order = "1_23")
        pl_tc_b, pl_t_c, slptc =
            merge_projectors_inter(M.lp, p_lb, p_lt, p_lc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (srt, srpt * srpcb))
        tmp2 = alloc_undef(R, onGPU, (slt * slpt, srpcb))
        tmp3 = alloc_zeros(R, onGPU, (slt * slpt, srpc * srpb))
        tmp5 = alloc_undef(R, onGPU, (slt * slpt, slpc, srpb))
        tmp8 = alloc_undef(R, onGPU, (slt * slptc, slpb))

        for irb ∈ 1:srb
            tmp1[:, pr_t_cb] = RE[irb, :, :]  # [rt, (rpt, rpcb)]
            mul!(tmp2, A2, reshape(tmp1, (srt * srpt, srpcb)))  # [(lt, lpt), rpcb]
            tmp3[:, pr_c_b] = tmp2  # [(lt, lpt), (rpc, rpb)]
            tmp4 = reshape(tmp3, (slt * slpt, srpc, srpb))  # [(lt, lpt), rpc, rpb]
            batched_mul!(tmp5, tmp4, M.con')  # [(lt, lpt), lpc, rpb]
            tmp6 = reshape(tmp5, (slt, slpt * slpc, srpb))  # [lt, (lpt, lpc), rpb]
            tmp7 = reshape(tmp6[:, pl_t_c, :], (slt * slptc, srpb))  # [(lb, lptc), rpb]
            for ilb ∈ 1:slb

                mul!(tmp8, tmp7, B[ilb, irb, :, :]')
                tmp9 = reshape(tmp8, (slt, slptc * slpb))
                Rout[ilb, :, :] .+= tmp9[:, pl_tc_b]
            end
        end
    end
    Rout
end

function update_reduced_env_right(
    K::Tensor{R,1},
    RE::Tensor{R,2},
    M::VirtualTensor{R,4},
    B::Tensor{R,3},
) where {R<:Real}
    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slpb, slpc, slpt = size(M.lp, p_lb), size(M.lp, p_lc), size(M.lp, p_lt)
    srpb, srpc, srpt = size(M.lp, p_rb), size(M.lp, p_rc), size(M.lp, p_rt)

    onGPU = typeof(RE) <: CuArray

    K = reshape(K, (slpt, srpt))  # [lct, rct]
    B = reshape(B, (slb, srb, slpb, srpb))  # [lb, rb, lpb, rpb]
    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lpb, rb, rpb]
    B2 = reshape(B2, (slb * slpb, srb * srpb))  # [(lb, lpb), (rb, rpb)]

    if srpb * slpt >= srpt * slpb
        pr_b_ct, pr_c_t, srpct =
            merge_projectors_inter(M.lp, p_rb, p_rc, p_rt, onGPU; order = "1_23")
        pl_bc_t, pl_b_c, slpbc =
            merge_projectors_inter(M.lp, p_lt, p_lb, p_lc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (srb, srpb * srpct))
        tmp4 = alloc_zeros(R, onGPU, (slb * slpb, srpc * srpt))
        tmp6 = alloc_undef(R, onGPU, (slb * slpb, slpc, srpt))

        tmp1[:, pr_b_ct] = RE  # [rb, (rpb, rpct)]
        tmp2 = reshape(tmp1, (srb * srpb, srpct))  # [(rb, rpb), rpct]
        tmp3 = B2 * tmp2  # [(lb, lpb), rpct]
        tmp4[:, pr_c_t] = tmp3
        tmp5 = reshape(tmp4, (slb * slpb, srpc, srpt))
        batched_mul!(tmp6, tmp5, M.con')  # [(lb, lpb), lpc, rpt]
        tmp7 = reshape(tmp6, (slb, slpb * slpc, srpt))
        tmp8 = tmp7[:, pl_b_c, :]  # [lb, lpbc, rpt]
        tmp9 = reshape(tmp8, (slb * slpbc, srpt)) * K' # [(lb, lpbc), lpt]
        tmp10 = reshape(tmp9, (slb, slpbc * slpt))
        Rtemp = tmp10[:, pl_bc_t]
    else
        pr_bc_t, pr_b_c, srpbc =
            merge_projectors_inter(M.lp, p_rt, p_rb, p_rc, onGPU; order = "23_1")
        pl_b_ct, pl_c_t, slpct =
            merge_projectors_inter(M.lp, p_lb, p_lc, p_lt, onGPU; order = "1_23")

        tmp1 = alloc_zeros(R, onGPU, (srb, srpbc * srpt))
        tmp4 = alloc_zeros(R, onGPU, (srb, srpb * srpc, slpt))
        tmp6 = alloc_undef(R, onGPU, (srb * srpb, slpc, slpt))

        tmp1[:, pr_bc_t] = RE  # [rb, (rpbc, rpt)]
        tmp2 = reshape(tmp1, (srb * srpbc, srpt))  # [(rb, rpbc), rpt]
        tmp3 = reshape(tmp2 * K', (srb, srpbc, slpt)) # [rb, rpbc, lpt]
        tmp4[:, pr_b_c, :] = tmp3
        tmp5 = reshape(tmp4, (srb * srpb, srpc, slpt))
        batched_mul!(tmp6, tmp5, M.con')  # [(rb, rpb), lpc, lpt]
        tmp7 = reshape(tmp6, (srb * srpb, slpc * slpt))  # [(rb, rpb), (lpc, lpt)]
        tmp8 = tmp7[:, pl_c_t]
        tmp9 = B2 * tmp8  # [(lb, lpb), lpct]
        tmp10 = reshape(tmp9, (slb, slpb * slpct))
        Rtemp = tmp10[:, pl_b_ct]
    end
    Rtemp
end


function contract_tensors43(M::VirtualTensor{R,4}, B::Tensor{R,3}) where {R<:Real}
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    slcp, srcp = length(M.lp, p_l), length(M.lp, p_r)

    B = reshape(B, (slb, srb, slcb, srcb))

    pls = sparse(R, M.lp, p_lb, p_l, p_lt, :CPU)
    pls = typeof(B) <: CuArray ? CuArray(pls) : Array(pls)
    pls = reshape(pls, (slcb, slc, slct * slcp))
    pls = permutedims(pls, (3, 1, 2))  # [(slct, slcp), lcb, lc]

    prs = sparse(R, M.lp, p_rb, p_r, p_rt, :CPU)
    prs = typeof(B) <: CuArray ? CuArray(prs) : Array(prs)
    prs = reshape(prs, (srcb, src, srct * srcp))
    prs = permutedims(prs, (3, 1, 2))  # [(rct, rcp), rcb, rc]

    if size(M.con, 1) <= size(M.con, 2)
        prs = contract_matrix_tensor3(M.con, prs)
    else
        pls = contract_tensor3_matrix(pls, M.con)
    end
    @tensor order = (lb, c, rb) MB[l, lt, r, rt] :=
        pls[lt, lb, c] * prs[rt, rb, c] * B[l, r, lb, rb]
    MB = reshape(MB, slb, slct, slcp, srb, srct, srcp)
    MB = permutedims(MB, (1, 3, 4, 6, 2, 5))
    reshape(MB, (slb * slcp, srb * srcp, slct * srct))
end

function corner_matrix(
    C::S,
    M::T,
    B::S,
) where {S<:Tensor{R,3},T<:VirtualTensor{R,4}} where {R<:Real}
    slb, srb = size(B, 1), size(B, 2)
    srcc, stc = size(C, 2), size(C, 3)
    V = contract_tensors43(M, B)
    vl, vr, vt = size(V, 1), size(V, 2), size(V, 3)
    V = reshape(V, (vl, srb, stc, vt))
    @tensor Cnew[vl, vt, vrr] := V[vl, srb, stc, vt] * C[srb, vrr, stc]
    reshape(Cnew, (slb, :, srcc, vt))
end
