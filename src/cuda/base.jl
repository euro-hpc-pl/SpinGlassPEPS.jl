for (T, N) ∈ ((:MPO, 4), (:MPS, 3))
    CuT = Symbol(:Cu, T)
    AT = Symbol(:Abstract, T)
    @eval begin
        export $CuT, $T

        struct $CuT{T <: Number} <: $AT{T}
            tensors::Vector{CuArray{T, $N}}
        end

        $CuT(::Type{T}, L::Int) where {T} = $CuT(Vector{CuArray{T, $N}}(undef, L))
        $CuT(L::Int) = $CuT(Float32, L)

        Base.copy(a::$CuT) = $CuT(copy(a.tensors))

        $T(A::$CuT) = $T([Array(arr) for arr in A.tensors])
        $CuT(A::$T) = $CuT([cu(arr) for arr in A.tensors])
    end
end
@inline CuMPS(A::AbstractArray) = CuMPS(cu(A), :right)
@inline CuMPS(A::AbstractArray, s::Symbol, args...) = CuMPS(cu(A), Val(s), typemax(Int), args...)
@inline CuMPS(A::AbstractArray, s::Symbol, Dcut::Int, args...) = CuMPS(cu(A), Val(s), Dcut, args...)
@inline CuMPS(A::AbstractArray, ::Val{:right}, Dcut::Int, args...) = _left_sweep_SVD(CuMPS, cu(A), Dcut, args...)
@inline CuMPS(A::AbstractArray, ::Val{:left}, Dcut::Int, args...) = _right_sweep_SVD(CuMPS, cu(A), Dcut, args...)

function CUDA.randn(::Type{CuMPS{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = CuMPS(L)
    ψ[1] = CUDA.randn(T, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = CUDA.randn(T, D, d, D)
    end
    ψ[end] = CUDA.randn(T, D, d, 1)
    ψ
end

CUDA.randn(::Type{CuMPS}, args...) = CUDA.randn(CuMPS{Float32}, args...)

function CUDA.randn(::Type{CuMPO{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = CUDA.randn(CuMPS{T}, L, D, d^2)
    CuMPO(ψ)
end

CUDA.randn(::Type{CuMPO}, args...) = CUDA.randn(CuMPO{Float32}, args...)

CuMPS(states::Vector{CuVector{T}}) where {T <: Number} = CuMPS(
    [reshape(copy(v), (1, length(v), 1)) for v ∈ states]
)


function is_left_normalized(ψ::CuMPS)
    for i ∈ eachindex(ψ)
        A = ψ[i]
        DD = size(A, 3)

        @cutensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
        if norm(cu(I(DD)) - Id) > 1e-6 return false end
    end
    true
end

function is_right_normalized(ϕ::CuMPS)
    for i ∈ eachindex(ϕ)
        B = ϕ[i]
        DD = size(B, 1)

        @cutensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        if norm(cu(I(DD)) - Id) > 1e-6 return false end
    end
    true
end

function _truncate_svd(F::CuSVD, Dcut::Int)
    U, Σ, V = F
    c = min(Dcut, size(U, 2))
    U = U[:, 1:c]
    Σ = Σ[1:c]
    V = V[:, 1:c]
    U, Σ, V
end

function _truncate_qr(F::CuQR, Dcut::Int)
    Q, R = F
    Q = CuMatrix(Q)
    c = min(Dcut, size(Q, 2))
    Q = Q[:, 1:c]
    R = R[1:c, :]
    _qr_fix!(Q, R)
end

LinearAlgebra.svd(M::CuMatrix, Dcut::Int, args...) = _truncate_svd(CUDA.svd(M), Dcut)
