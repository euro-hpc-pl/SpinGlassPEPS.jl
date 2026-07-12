# site.jl: contractions with SiteTensor on CPU and CUDA

# TODO make sure slicing is done right,
# cf. https://discourse.julialang.org/t/correct-implementation-of-cuarrays-slicing-operations/90600

function contract_sparse_with_three(
    lp,
    X1::S,
    X2::S,
    X3::S,
    loc_exp::T,
    k1::Q,
    k2::Q,
    k3::Q,
    kout::Q,
) where {S<:Tensor{R,3},T<:Tensor{R,1},Q<:Integer} where {R<:Real}
    s1, s2, _ = size(X1)
    s3, s4, _ = size(X3)

    device = typeof(loc_exp) <: CuArray ? :GPU : :CPU
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, device)
    p3 = get_projector!(lp, k3, device)

    total_memory = 2^32 # TODO add better handling for this; also depending on device
    batch_size = max(
        Int(
            floor(
                total_memory /
                (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1 + min(s1 * s3, s2 * s4))),
            ),
        ),
        1,
    )
    batch_size = Int(2^floor(log2(batch_size) + 1e-6))

    total_size = length(p1)
    batch_size = min(batch_size, total_size)

    onGPU = typeof(loc_exp) <: CuArray
    out = onGPU ? CUDA.zeros(R, size(lp, kout), s1, s4) : zeros(R, size(lp, kout), s1, s4)

    from = 1
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        vp1 = @view p1[from:to]
        vp2 = @view p2[from:to]
        vp3 = @view p3[from:to]

        X1p = X1[:, :, vp1]
        X2p = X2[:, :, vp2]
        X3p = X3[:, :, vp3]

        if s1 * s3 < s2 * s4
            Xtmp = batched_mul(X1p, X2p)
            outp = batched_mul(Xtmp, X3p)
        else
            Xtmp = batched_mul(X2p, X3p)
            outp = batched_mul(X1p, Xtmp)
        end

        le = @view loc_exp[from:to]
        outp .*= reshape(le, 1, 1, :)
        outpp = reshape(outp, s1 * s4, :)
        ipr, rf, rt = sparse(R, lp, kout, device; from, to)
        @inbounds out[rf:rt, :, :] .+= reshape(ipr * outpp', :, s1, s4)
        from = to + 1
    end
    permutedims(out, (2, 3, 1))
end

function update_env_left(
    LE::S,
    A::S,
    M::T,
    B::S,
) where {S<:Tensor{R,3},T<:SiteTensor{R,4}} where {R<:Real}
    contract_sparse_with_three(
        M.lp,
        permutedims(B, (2, 1, 3)),
        LE,
        A,
        M.loc_exp,
        M.projs[[4, 1, 2, 3]]...,
    )
end

function update_env_right(
    RE::S,
    A::S,
    M::SiteTensor{R,4},
    B::S,
) where {S<:Tensor{R,3}} where {R<:Real}
    contract_sparse_with_three(
        M.lp,
        B,
        RE,
        permutedims(A, (2, 1, 3)),
        M.loc_exp,
        M.projs[[4, 3, 2, 1]]...,
    )
end

function project_ket_on_bra(
    LE::S,
    B::S,
    M::SiteTensor{R,4},
    RE::S,
) where {S<:Tensor{R,3}} where {R<:Real}
    contract_sparse_with_three(
        M.lp,
        permutedims(LE, (2, 1, 3)),
        B,
        RE,
        M.loc_exp,
        M.projs[[1, 4, 3, 2]]...,
    )
end

function update_reduced_env_right(
    K::Tensor{R,1},
    RE::Tensor{R,2},
    M::SiteTensor{R,4},
    B::Tensor{R,3},
) where {R<:Real}
    device = typeof(M.loc_exp) <: CuArray ? :GPU : :CPU
    s1, s2, _ = size(B)

    p2, p3, p4 = (get_projector!(M.lp, x, device) for x in M.projs[2:4])
    k1 = M.projs[1]
    total_memory = 2^32 # TODO add better handling for this; also depending on device

    batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s1 + s2 + 1)))), 1)
    batch_size = Int(2^floor(log2(batch_size) + 1e-6))

    out =
        typeof(M.loc_exp) <: CuArray ? CUDA.zeros(R, size(M.lp, k1), s1) :
        zeros(R, size(M.lp, k1), s1)
    RE = reshape(RE, size(RE, 1), 1, size(RE, 2))

    from = 1
    total_size = length(p4)
    while from <= total_size
        to = min(total_size, from + batch_size - 1)
        vp2 = @view p2[from:to]
        vp3 = @view p3[from:to]
        vp4 = @view p4[from:to]

        @inbounds Kp = K[vp2]
        @inbounds REp = RE[:, :, vp3]
        @inbounds Bp = B[:, :, vp4]
        le = @view M.loc_exp[from:to]

        outp = dropdims(Bp ⊠ REp, dims = 2)
        outp .*= reshape(le .* Kp, 1, :)

        ipr, rf, rt = sparse(R, M.lp, k1, device; from, to)
        @inbounds out[rf:rt, :] .+= ipr * outp'
        from = to + 1
    end
    permutedims(out, (2, 1))
end

function contract_tensors43(M::SiteTensor{R,4}, B::Tensor{R,3}) where {R<:Real}
    device = typeof(M.loc_exp) <: CuArray ? :GPU : :CPU
    p4 = get_projector!(M.lp, M.projs[4], device)
    sb1, sb2, _ = size(B)
    sm1, sm2, sm3 = size.(Ref(M.lp), M.projs[1:3])
    @inbounds Bp = B[:, :, p4] .* reshape(M.loc_exp, 1, 1, :)
    # @cast Bp[(x, y), z] := Bp[x, y, z] TODO: restore when deps merged
    Bp = reshape(Bp, size(Bp, 1) * size(Bp, 2), :)
    ip123 = sparse(R, M.lp, M.projs[1], M.projs[2], M.projs[3], device)
    out = reshape(ip123 * Bp', sm1, sm2, sm3, sb1, sb2)
    out = permutedims(out, (4, 1, 5, 3, 2))
    reshape(out, sb1 * sm1, sb2 * sm3, sm2)
end

function corner_matrix(
    C::S,
    M::T,
    B::S,
) where {S<:Tensor{R,3},T<:SiteTensor{R,4}} where {R<:Real}
    device = typeof(M.loc_exp) <: CuArray ? :GPU : :CPU
    projs = [get_projector!(M.lp, x, device) for x in M.projs]
    @inbounds Bp = B[:, :, projs[4]]
    @inbounds Cp = C[:, :, projs[3]]
    outp = Bp ⊠ Cp
    outp .*= reshape(M.loc_exp, 1, 1, :)
    # @cast outp[(x, y), z] := outp[x, y, z]  TODO: restore when deps merged
    Bp = reshape(outp, size(Bp, 1) * size(outp, 2), :)
    sm1, sm2 = maximum(projs[1]), maximum(projs[2])
    @inbounds p12 = projs[1] .+ (projs[2] .- 1) .* sm1
    ip12 = sparse(R, p12; mp = sm1 * sm2)
    out = reshape(ip12 * outp', sm1, maximum(projs[2]), size(B, 1), size(C, 2))
    permutedims(out, (3, 1, 4, 2))
end
