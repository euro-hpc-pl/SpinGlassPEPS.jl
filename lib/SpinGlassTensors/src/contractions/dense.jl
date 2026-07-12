# contractions of dense objects on CPU and CUDA
# export
#     update_reduced_env_right2

const MatrixOrCuMatrix{R} = Union{
    CuMatrix{R},
    Matrix{R},
    Diagonal{R,CuArray{R,1,CUDA.DeviceMemory}},
    Diagonal{R,Vector{R}},
}

function contract_tensor3_matrix(A::Tensor{R,3}, M::MatrixOrCuMatrix{R}) where {R<:Real}
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M, sl1, sl2, :)
end

function contract_matrix_tensor3(M::MatrixOrCuMatrix{R}, A::Tensor{R,3}) where {R<:Real}
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M', sl1, sl2, :)
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S,
    A::S,
    M::T,
    B::S,
) where {S<:Tensor{R,3},T<:Tensor{R,4}} where {R<:Real}
    @tensor order = (ot, α, oc, β, ob) LE[nb, nt, nc] :=
        LE[ob, ot, oc] * A[ot, nt, α] * M[oc, α, nc, β] * B[ob, nb, β] # TODO: split the line
end

"""
        -- A --
      |    |
 L = LE    |
      |    |
        -- B --
"""
function update_env_left(
    LE::T,
    A::S,
    B::S,
) where {S<:Tensor{R,3},T<:Tensor{R,2}} where {R<:Real}
    @tensor order = (ot, α, ob) LE[nb, nt] := LE[ob, ot] * A[ot, nt, α] * B[ob, nb, α]
end

"""
        -- A --
      |    |
 L = LE
      |

"""
function update_env_left(LE::T, A::S) where {S<:Tensor{R,3},T<:Tensor{R,2}} where {R<:Real}
    @tensor A[nb, nt, nc] := LE[nb, ot] * A[ot, nt, nc]
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S,
    A::S,
    M::T,
    B::S,
) where {T<:Tensor{R,4},S<:Tensor{R,3}} where {R<:Real}
    @tensor order = (ot, α, oc, β, ob) RE[nb, nt, nc] :=
        RE[ob, ot, oc] * A[nt, ot, α] * M[nc, α, oc, β] * B[nb, ob, β]
end

"""
      -- A --
         |    |
 R =     |    RE
         |    |
      -- B --
"""
function update_env_right(
    RE::T,
    A::S,
    B::S,
) where {T<:Tensor{R,2},S<:Tensor{R,3}} where {R<:Real}
    @tensor order = (ot, α, ob) RE[nb, nt] := RE[ob, ot] * A[nt, ot, α] * B[nb, ob, α]
end

"""
      -- A --
         |    |
 R =      --- RE
              |

"""
function update_env_right(RE::S, C::S) where {S<:Tensor{R,3}} where {R<:Real}
    @tensor order = (ot, oc) RR[nb, nt] := RE[nb, ot, oc] * C[nt, ot, oc]
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S,
    B::S,
    M::T,
    RE::S,
) where {T<:Tensor{R,4},S<:Tensor{R,3}} where {R<:Real}
    @tensor order = (ol, lc, oc, or, rc) A[nl, nr, nc] :=
        LE[ol, nl, lc] * B[ol, or, oc] * M[lc, nc, rc, oc] * RE[or, nr, rc]
end

"""
  LE -     - RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::T,
    B::S,
    RE::T,
) where {T<:Tensor{R,2},S<:Tensor{R,3}} where {R<:Real}
    @tensor order = (ol, or) A[nl, nr, nc] := LE[ol, nl] * B[ol, or, nc] * RE[or, nr]
end

"""
   |      |
  LE ---- RE --
"""
function project_ket_on_bra(
    LE::T,
    RE::S,
) where {T<:Tensor{R,2},S<:Tensor{R,3}} where {R<:Real}
    @tensor A[nl, nr, nc] := LE[ol, nl] * RE[ol, nr, nc]
end

"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(
    RE::Tensor{R,2},
    m::Int,
    M::MpoTensor{R,4},
    B::Tensor{R,3},
) where {R<:Real}
    K = zeros(R, size(M, 2))
    K[m] = one(R)
    if typeof(RE) <: CuArray
        K = CuArray(K)
    end
    K = reshape(K, 1, 1, size(K, 1))
    for v ∈ M.top
        K = contract_tensor3_matrix(K, v)
    end
    K = dropdims(K, dims = (1, 2))

    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)   # TODO do we ever enter here? in mpo layers that we have now, we don't
    end
    update_reduced_env_right(K, RE, M.ctr, B)
end

function update_reduced_env_right(
    K::Tensor{R,1},
    RE::Tensor{R,2},
    M::Tensor{R,4},
    B::Tensor{R,3},
) where {R<:Real}
    @tensor order = (d, β, γ, α) RE[x, y] := K[d] * M[y, d, β, γ] * B[x, α, γ] * RE[α, β]
end

function update_reduced_env_right(RR::S, M0::S) where {S<:Tensor{<:Real,2}}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function contract_tensors43(B::Tensor{R,4}, A::Tensor{R,3}) where {R<:Real}
    # @matmul C[(x, y), (b, a), z] := sum(σ) B[y, z, a, σ] * A[x, b, σ]
    @tensor C[x, y, b, a, z] := B[y, z, a, σ] * A[x, b, σ]
    C = reshape(C, size(C, 1) * size(C, 2), size(C, 3) * size(C, 4), size(C, 5))
    return C
end

function corner_matrix(
    C::S,
    M::T,
    B::S,
) where {S<:Tensor{R,3},T<:Tensor{R,4}} where {R<:Real}
    @tensor order = (rr, mb, mr) Cnew[ll, ml, tt, mt] :=
        M[ml, mt, mr, mb] * B[ll, rr, mb] * C[rr, tt, mr]
end
