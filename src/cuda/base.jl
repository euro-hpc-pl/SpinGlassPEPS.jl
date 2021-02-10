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

        function $T(A::$CuT)
            T = eltype(A)
            L = length(A)
            devA = $T(T, L)
            for i ∈ eachindex(A)
                devA[i] = Array(A[i])
            end
            devA
        end

        function $CuT(A::$T)
            T = eltype(A)
            L = length(A)
            cuA = $CuT(T, L)
            for i ∈ eachindex(A)
                cuA[i] = cu(A[i])
            end
            cuA
        end
    end
end
@inline CuMPS(A::AbstractArray) = CuMPS(cu(A), :right)
@inline CuMPS(A::AbstractArray, s::Symbol, args...) = CuMPS(cu(A), Val(s), typemax(Int), args...)
@inline CuMPS(A::AbstractArray, s::Symbol, Dcut::Int, args...) = CuMPS(cu(A), Val(s), Dcut, args...)
@inline CuMPS(A::AbstractArray, ::Val{:right}, Dcut::Int, args...) = _left_sweep_SVD(cu(A), Dcut, args...)
@inline CuMPS(A::AbstractArray, ::Val{:left}, Dcut::Int, args...) = _right_sweep_SVD(cu(A), Dcut, args...)

function CUDA.randn(::Type{CuMPS{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = CuMPS(L)
    ψ[1] = CUDA.randn(T, 1, d, D)
    for i ∈ 2:(L-1)
        ψ[i] = CUDA.randn(T, D, d, D)
    end
    ψ[end] = CUDA.randn(T, D, d, 1)
    ψ
end

function CUDA.randn(::Type{CuMPO{T}}, L::Int, D::Int, d::Int) where {T}
    ψ = CUDA.randn(CuMPS{T}, L, D, d^2) 
    CuMPO(ψ)
end

function CuMPS(vec::CuVector{<:Number})
    L = length(vec)
    ψ = CuMPS(L)
    for i ∈ 1:L
        ψ[i] = reshape(copy(vec[i]), 1, :, 1)
    end    
end

function CuMPO(ψ::CuMPS)
    _verify_square(ψ)
    L = length(ψ)
    O = CuMPO(eltype(ψ), L)

    for i ∈ 1:L
        A = ψ[i]
        d = isqrt(size(A, 2))
        
        @cast W[x, σ, y, η] |= A[x, (σ, η), y] (σ:d) 
        O[i] = W
    end 
    O
end 

function CuMPS(O::CuMPO)
    L = length(O)
    ψ = CuMPS(eltype(O), L)

    for i ∈ 1:L
        W = O[i]
        @cast A[x, (σ, η), y] := W[x, σ, y, η]
        ψ[i] = A     
    end 
    return ψ
end

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

function _right_sweep_SVD(Θ::CuArray{T}, Dcut::Int=typemax(Int), args...) where {T}
    rank = ndims(Θ)
    ψ = CuMPS(T, rank)

    V = reshape(copy(conj(Θ)), (length(Θ), 1))

    for i ∈ 1:rank
        d = size(Θ, i)

        # reshape
        VV = conj.(transpose(V)) # hack @cast does not work with Adjoint
        @cast M[(x, σ), y] |= VV[x, (σ, y)] (σ:d)
       
        # decompose
        U, Σ, V = _truncate_svd(CUDA.svd(M), Dcut)
        V *= Diagonal(Σ)

        # create MPS  
        @cast A[x, σ, y] |= U[(x, σ), y] (σ:d)
        ψ[i] = A
    end
    ψ
end

function _left_sweep_SVD(Θ::CuArray{T}, Dcut::Int=typemax(Int), args...) where {T}
    rank = ndims(Θ)
    ψ = CuMPS(T, rank)

    U = reshape(copy(Θ), (length(Θ), 1))

    for i ∈ rank:-1:1
        d = size(Θ, i)

        # reshape
        @cast M[x, (σ, y)] |= U[(x, σ), y] (σ:d)

        # decompose
        U, Σ, V = _truncate_svd(CUDA.svd(M), Dcut)
        U *= Diagonal(Σ)

        # create MPS  
        VV = conj.(transpose(V)) # hack @cast does not work with Adjoint
        @cast B[x, σ, y] |= VV[x, (σ, y)] (σ:d)
        ψ[i] = B
    end
    ψ
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

# function tensor(state::State, tensors...)
#     C = I
#     for (A, σ) ∈ zip(tensors, state)
#         C *= A[:, idx(σ), :]
#     end
#     tr(C)
# end

# function tensor(ψ::CuMPS)
#     dims = rank(ψ)
#     T = eltype(ψ)
#     ret = CUDA.zeros(T, dims)
#     CI = CartesianIndices(ret)

#     @inline function kernel(ret, t1, t2)
#         state = (blockIdx().x-1) * blockDim().x + threadIdx().x
#         if state <= length(ret)
#             @inbounds idx = CI[state].I
#             σ = _σ.(idx)
#             @inbounds ret[state] = tensor(σ, t1, t2)
#         end
#         return
#     end

#     threads, blocks = cudiv(length(ret))
#     CUDA.@cuda threads=threads blocks=blocks kernel(ret, ψ.tensors[1], ψ.tensors[2])
#     ret
# end

function tensor(ψ::CuMPS)
    devψ = MPS(ψ)
    t = tensor(devψ)
    cu(t)
end