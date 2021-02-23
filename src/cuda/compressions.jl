function _right_sweep_SVD!(ψ::CuMPS, Dcut::Int=typemax(Int))
    Σ = V = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ eachindex(ψ)
        A = ψ[i]
        C = Diagonal(Σ) * V'

        # attach
        @cutensor M[x, σ, y] := C[x, α] * A[α, σ, y]
        @cast   M̃[(x, σ), y] |= M[x, σ, y]
        
        # decompose
        U, Σ, V = _truncate_svd(CUDA.svd(M̃), Dcut)

        # create new    
        d = physical_dim(ψ, i)
        @cast A[x, σ, y] |= U[(x, σ), y] (σ:d)
        ψ[i] = A
    end
end

function _left_sweep_SVD!(ψ::CuMPS, Dcut::Int=typemax(Int))
    Σ = U = CUDA.ones(eltype(ψ), 1, 1)

    for i ∈ length(ψ):-1:1
        B = ψ[i]
        C = U * Diagonal(Σ)

        # attach
        @cutensor M[x, σ, y]   := B[x, σ, α] * C[α, y]
        @cast   M̃[x, (σ, y)] |= M[x, σ, y]

        # decompose
        U, Σ, V = _truncate_svd(CUDA.svd(M̃), Dcut)

        # create new 
        d = physical_dim(ψ, i)
        VV = conj.(transpose(V)) #hack - @cast does fail with allowscalar and Adjoint type
        @cast B[x, σ, y] |= VV[x, (σ, y)] (σ:d)
        ψ[i] = B
    end
end

function _left_sweep_var!!(ϕ::CuMPS, env::Vector{<:CuMatrix}, ψ::CuMPS, Dcut::Int)
    S = eltype(ϕ)
    
    # overwrite the overlap
    env[end] = CUDA.ones(S, 1, 1)

    for i ∈ length(ψ):-1:1
        L = env[i]
        R = env[i+1]

        # optimize site
        M = ψ[i]
        @cutensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β) 
        
        # right canonize it
        @cast MM[x, (σ, y)] |= M̃[x, σ, y]
        Q = _truncate_qr(CUDA.qr(MM'), Dcut)'

        Q = CuMatrix(Q) #hack @cast does not work with Adjoint
        d = size(M, 2)
        @cast B[x, σ, y] |= Q[x, (σ, y)] (σ:d)

        # update ϕ and right environment 
        ϕ[i] = B
        A = ψ[i]

        @cutensor RR[x, y] := A[x, σ, α] * R[α, β] * conj(B[y, σ, β]) order = (β, α, σ)
        env[i] = RR
    end    
end

function _right_sweep_var!!(ϕ::CuMPS, env::Vector{<:CuMatrix}, ψ::CuMPS, Dcut::Int)
    S = eltype(ϕ)
    
    # overwrite the overlap
    env[1] = CUDA.ones(S, 1, 1)

    for i ∈ eachindex(ψ)
        L = env[i]
        R = env[i+1]

        # optimize site
        M = ψ[i]
        @cutensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β) 
                   
        # left canonize it
        @cast B[(x, σ), y] |= M̃[x, σ, y]
        Q = _truncate_qr(CUDA.qr(B), Dcut)

        d = size(ϕ[i], 2)
        @cast A[x, σ, y] |= Q[(x, σ), y] (σ:d)

        # update ϕ and left environment 
        ϕ[i] = A
        B = ψ[i]

        @cutensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end
    return real(tr(env[end]))
end

function _qr_fix!(Q::CuMatrix, R::CuMatrix)
    d = diag(R)
    ph = d./abs.(d)
    transpose(ph) .* Q
end
