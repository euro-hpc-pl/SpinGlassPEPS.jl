export zipper, corner_matrix, CornerTensor

struct CornerTensor{T<:Real}
    C::Tensor{T,3}
    M::MpoTensor{T,4}
    B::Tensor{T,3}

    function CornerTensor(C, M, B)
        T = promote_type(eltype.((C, M, B))...)
        new{T}(C, M, B)
    end
end

struct Adjoint{T,S<:CornerTensor}
    parent::S

    function Adjoint{T}(ten::CornerTensor{S}) where {T,S}
        F = promote_type(T, S)
        new{F,CornerTensor{F}}(ten)
    end
end

function zipper(
    ψ::QMpo{R},
    ϕ::QMps{R};
    method::Symbol = :svd,
    Dcut::Int = typemax(Int),
    tol = eps(),
    iters_rand = 3,
    iters_svd = 1,
    iters_var = 1,
    Dtemp_multiplier = 2,
    depth::Int = 0,
    kwargs...,
) where {R<:Real}
    onGPU = ψ.onGPU && ϕ.onGPU
    @assert is_left_normalized(ϕ)

    C = onGPU ? CUDA.ones(R, 1, 1, 1) : ones(R, 1, 1, 1)
    mpo_li = last(ψ.sites)

    d = (depth == 0) ? mpo_li : depth

    Dtemp = Dtemp_multiplier * Dcut
    out = copy(ϕ)
    env = EnvironmentMixed(out, C, ψ, ϕ)

    for i ∈ reverse(ϕ.sites)
        while mpo_li > i
            C = contract_matrix_tensor3(ψ[mpo_li], C)
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        @assert mpo_li == i "Mismatch between QMpo and QMps sites."
        mpo_li = left_nbrs_site(mpo_li, ψ.sites)

        if i > ϕ.sites[1]
            CM = CornerTensor(C, ψ[i], out[i])

            Urs, Srs, Vrs = [], [], []
            for i = 1:iters_rand
                Utemp, Stemp, Vtemp =
                    svd_corner_matrix(CM, method, Dtemp, tol; toGPU = false, kwargs...)
                push!(Urs, Utemp)
                push!(Srs, Stemp)
                push!(Vrs, Vtemp)
            end

            Ur = hcat(Urs...)
            Vr = hcat(Vrs...)
            Sr = vcat(Srs...) ./ iters_rand
            QU, RU = qr_fact(Ur, Dtemp * iters_rand, zero(R); toGPU = false, kwargs...)
            QV, RV = qr_fact(Vr, Dtemp * iters_rand, zero(R); toGPU = false, kwargs...)
            Ur, Sr, Vr = svd_fact(RU * Diagonal(Sr) * RV', Dtemp, tol; kwargs...)
            # Ur = QU * Ur
            Vr = QV * Vr

            if onGPU
                Vr = CuArray(Vr)
            end

            for _ = 1:iters_svd
                # CM * Vr
                x = reshape(Vr, size(CM.C, 2), size(CM.M, 2), :)
                x = permutedims(x, (3, 1, 2))
                x = update_env_right(CM.C, x, CM.M, CM.B)
                CCC = reshape(permutedims(x, (1, 3, 2)), size(CM.B, 1) * size(CM.M, 1), :)

                Ut, _ = qr_fact(CCC, Dtemp, zero(R); toGPU = ψ.onGPU, kwargs...)

                # CM' * Ut
                x = reshape(Ut, size(CM.B, 1), size(CM.M, 1), :)
                x = permutedims(x, (1, 3, 2))
                yp = project_ket_on_bra(x, CM.B, CM.M, CM.C)
                CCC = reshape(permutedims(yp, (2, 3, 1)), size(CM.C, 2) * size(CM.M, 2), :)

                Vr, _ = qr_fact(CCC, Dtemp, zero(R); toGPU = ψ.onGPU, kwargs...)
            end

            # CM * Vr
            x = reshape(Vr, size(CM.C, 2), size(CM.M, 2), :)
            x = permutedims(x, (3, 1, 2))
            x = update_env_right(CM.C, x, CM.M, CM.B)
            CCC = reshape(permutedims(x, (1, 3, 2)), size(CM.B, 1) * size(CM.M, 1), :)
            CCC ./= norm(CCC)

            V, CCC = qr_fact(CCC', Dcut, tol; toGPU = ψ.onGPU, kwargs...)
            V = V' * Vr'
            s1, s2 = size(ψ[i])
            # @cast CCC[z, x, y] := CCC[z, (x, y)] (y ∈ 1:s1)
            CCC = reshape(CCC, size(CCC, 1), size(CCC, 2) ÷ s1, s1)
            C = permutedims(CCC, (2, 1, 3))
            # @cast V[x, y, z] := V[x, (y, z)] (z ∈ 1:s2)
            V = reshape(V, size(V, 1), size(V, 2) ÷ s2, s2)
            out[i] = V
        else
            L = onGPU ? CUDA.ones(R, 1, 1, 1) : ones(R, 1, 1, 1)
            V = project_ket_on_bra(L, out[i], ψ[i], C)
            V ./= norm(V)
            out[i] = V
            C = onGPU ? CUDA.ones(R, 1, 1, 1) : ones(R, 1, 1, 1)
        end

        for _ = 1:iters_var
            env.site = i
            update_env_right!(env, i)
            env.C = C
            update_env_left!(env, :central)
            _left_sweep_var_site!(env, :central; kwargs...)
            for k in reverse(ϕ.sites)
                if (i - d) <= k < i
                    _left_sweep_var_site!(env, k; kwargs...)
                end
            end
            for k in ϕ.sites
                if (i - d) <= k < i
                    _right_sweep_var_site!(env, k; kwargs...)
                end
            end
            _right_sweep_var_site!(env, :central; kwargs...)

            for k in ϕ.sites
                if (i + d) >= k >= i
                    _right_sweep_var_site!(env, k; kwargs...)
                end
            end
            for k in reverse(ϕ.sites)
                if (i + d) >= k >= i
                    _left_sweep_var_site!(env, k; kwargs...)
                end
            end
            update_env_right!(env, :central)
            C = project_ket_on_bra(env, :central)
        end
    end
    out
end

function _left_sweep_var_site!(env::EnvironmentMixed, site; kwargs...)   # site: Union(Sites, :central)
    update_env_right!(env, site)
    A = project_ket_on_bra(env, site)
    # @cast B[l, (r, t)] := A[l, r, t]
    B = reshape(A, size(A, 1), size(A, 2) * size(A, 3))
    _, Q = rq_fact(B; toGPU = env.onGPU, kwargs...)
    # @cast C[l, r, t] := Q[l, (r, t)] (t ∈ 1:size(A, 3))
    C = reshape(Q, size(Q, 1), size(Q, 2) ÷ size(A, 3), size(A, 3))
    !env.onGPU && (C = collect(C))
    if site == :central
        env.C = C
    else
        env.bra[site] = C
    end
    clear_env_containing_site!(env, site)
end

function _right_sweep_var_site!(env::EnvironmentMixed, site; kwargs...)
    update_env_left!(env, site)
    A = project_ket_on_bra(env, site)
    B = permutedims(A, (1, 3, 2))  # [l, t, r]
    # @cast B[(l, t), r] := B[l, t, r]
    B = reshape(B, size(B, 1) * size(B, 2), size(B, 3))
    Q, _ = qr_fact(B; toGPU = env.onGPU, kwargs...)
    # @cast C[l, t, r] := Q[(l, t), r] (t ∈ 1:size(A, 3))
    C = reshape(Q, size(Q, 1) ÷ size(A, 3), size(A, 3), size(Q, 2))
    C = permutedims(C, (1, 3, 2))  # [l, r, t]
    !env.onGPU && (C = collect(C))
    if site == :central
        env.C = C
    else
        env.bra[site] = C
    end
    clear_env_containing_site!(env, site)
end

function Base.Array(CM::CornerTensor)  # change name, or be happy with "psvd(Array(Array(CM))"
    B, M, C = CM.B, CM.M, CM.C
    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)
    end
    Cnew = corner_matrix(C, M.ctr, B)
    # @cast Cnew[(t1, t2), t3, t4] := Cnew[t1, t2, t3, t4]
    Cnew = reshape(Cnew, size(Cnew, 1) * size(Cnew, 2), size(Cnew, 3), size(Cnew, 4))
    for v ∈ reverse(M.top)
        Cnew = contract_matrix_tensor3(v, Cnew)
    end
    # @cast Cnew[t12, (t3, t4)] := Cnew[t12, t3, t4]
    Cnew = reshape(Cnew, size(Cnew, 1), size(Cnew, 2) * size(Cnew, 3))
end

Base.Array(CM::Adjoint{T,CornerTensor{T}}) where {T} = adjoint(Array(CM.ten))

# TODO rethink this function
function svd_corner_matrix(
    CM,
    method::Symbol,
    Dcut::Int,
    tol::Real;
    toGPU::Bool = true,
    kwargs...,
)
    if method == :svd
        U, Σ, V = svd_fact(Array(Array(CM)), Dcut, tol; kwargs...)
    elseif method == :psvd
        U, Σ, V = psvd(Array(Array(CM)), rank = Dcut)
    elseif method == :psvd_sparse
        U, Σ, V = psvd(CM, rank = Dcut)
    elseif method == :tsvd
        U, Σ, V = tsvd(Array(CM), Dcut; kwargs...)
    elseif method == :tsvd_sparse
        v0 = 2 .* rand(eltype(CM), size(CM, 1)) .- 1
        U, Σ, V = tsvd(CM, Dcut, initvec = v0; kwargs...)
    else
        throw(ArgumentError("Wrong method $method"))
    end
    toGPU && return CuArray.((U, Σ, V))
    U, Σ, V
end

# this is for psvd to work
LinearAlgebra.ishermitian(ten::CornerTensor) = false
Base.size(ten::CornerTensor) =
    (size(ten.B, 1) * size(ten.M, 1), size(ten.C, 2) * size(ten.M, 2))
Base.size(ten::CornerTensor, n::Int) = size(ten)[n]
Base.eltype(ten::CornerTensor{T}) where {T} = T
Base.adjoint(ten::CornerTensor{T}) where {T} = Adjoint{T}(ten)

CuArrayifneeded(ten::CornerTensor, x) = typeof(ten.B) <: CuArray ? CuArray(x) : x
CuArrayifneeded(ten::Adjoint{T,CornerTensor{T}}, x) where {T} =
    CuArrayifneeded(ten.parent, x)


function LinearAlgebra.mul!(y, ten::CornerTensor, x)
    x = CuArrayifneeded(ten, x) # CuArray(x) # TODO this an ugly patch
    x = reshape(x, size(ten.C, 2), size(ten.M, 2), :)
    x = permutedims(x, (3, 1, 2))
    yp = update_env_right(ten.C, x, ten.M, ten.B)
    y[:, :] .=
        Array(reshape(permutedims(yp, (1, 3, 2)), size(ten.B, 1) * size(ten.M, 1), :))
end

function LinearAlgebra.mul!(y, ten::Adjoint{T,CornerTensor{T}}, x) where {T<:Real}
    x = CuArrayifneeded(ten, x)  # CuArray(x)  # TODO this an ugly patch
    x = reshape(x, size(ten.parent.B, 1), size(ten.parent.M, 1), :)
    x = permutedims(x, (1, 3, 2))
    yp = project_ket_on_bra(x, ten.parent.B, ten.parent.M, ten.parent.C)
    y[:, :] .= Array(
        reshape(
            permutedims(yp, (2, 3, 1)),
            size(ten.parent.C, 2) * size(ten.parent.M, 2),
            :,
        ),
    )
end

function Base.:(*)(ten::CornerTensor{T}, x) where {T}
    x = CuArrayifneeded(ten, x)  # CuArray(x)  # TODO this an ugly patch
    x = reshape(x, 1, size(ten.C, 2), size(ten.M, 2))
    yp = update_env_right(ten.C, x, ten.M, ten.B)
    out = reshape(yp, size(ten.B, 1) * size(ten.M, 1))
    Array(out)  # TODO this an ugly patch
end

function Base.:(*)(ten::Adjoint{T,CornerTensor{T}}, x) where {T<:Real}
    x = CuArrayifneeded(ten, x)  # CuArray(x)  # TODO this an ugly patch
    x = reshape(x, size(ten.parent.B, 1), 1, size(ten.parent.M, 1))
    yp = project_ket_on_bra(x, ten.parent.B, ten.parent.M, ten.parent.C)
    out = reshape(yp, size(ten.parent.C, 2) * size(ten.parent.M, 2))
    Array(out)  # TODO this an ugly patch
end
