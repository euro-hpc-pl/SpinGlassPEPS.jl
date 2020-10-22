export truncate!, canonise!, compress

# TODO
# 1) write two sites compress function

function LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int) 
    fact = pqrfact(M, rank=Dcut)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix!(Q, R)
end     

function rq(M::AbstractMatrix, Dcut::Int) 
    fact = pqrfact(:c, conj(M), rank=Dcut)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix!(Q, R)'
end  

function _qr_fix!(Q::AbstractMatrix, R::AbstractMatrix)
    d = diag(R)
    ph = d./abs.(d)
    idim = size(R, 1)
    q = Matrix(Q)[:, 1:idim]
    return transpose(ph) .* q
end

function canonise!(ψ::MPS)
    canonise!(ψ, :right)
    canonise!(ψ, :left)
end

canonise!(ψ::MPS, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::MPS, ::Val{:right}) = _left_sweep_SVD!(ψ)
canonise!(ψ::MPS, ::Val{:left}) = _right_sweep_SVD!(ψ)

truncate!(ψ::MPS, s::Symbol, Dcut::Int) = truncate!(ψ, Val(s), Dcut)
truncate!(ψ::MPS, ::Val{:right}, Dcut::Int) = _left_sweep_SVD!(ψ, Dcut)
truncate!(ψ::MPS, ::Val{:left}, Dcut::Int) = _right_sweep_SVD!(ψ, Dcut)

function _right_sweep_SVD!(ψ::MPS, Dcut::Int=typemax(Int))
    Σ = V = ones(eltype(ψ), 1, 1)

    for i ∈ 1:length(ψ)
        A = ψ[i]
        C = Diagonal(Σ) * V'

        # attach
        @tensor M[x, σ, y] := C[x, α] * A[α, σ, y]
        @cast   M̃[(x, σ), y] |= M[x, σ, y]
        
        # decompose
        U, Σ, V = psvd(M̃, rank=Dcut)

        # create new    
        d = size(ψ[i], 2)
        @cast A[x, σ, y] |= U[(x, σ), y] (σ:d)
        ψ[i] = A
    end
end

function _left_sweep_SVD!(ψ::MPS, Dcut::Int=typemax(Int))
    Σ = U = ones(eltype(ψ), 1, 1)

    for i ∈ length(ψ):-1:1
        B = ψ[i]
        C = U * Diagonal(Σ)

        # attach
        @tensor M[x, σ, y]   := B[x, σ, α] * C[α, y]
        @cast   M̃[x, (σ, y)] |= M[x, σ, y]

        # decompose
        U, Σ, V = psvd(M̃, rank=Dcut)

        # create new 
        d = size(ψ[i], 2)
        @cast B[x, σ, y] |= V'[x, (σ, y)] (σ:d)
        ψ[i] = B
    end
end
 

function compress(ψ::MPS, Dcut::Int, tol::Number, max_sweeps::Int=4)

    # Initial guess - truncated ψ 
    ϕ = copy(ψ)
    truncate!(ϕ, :right, Dcut)

    # Create environment 
    env = left_env(ϕ, ψ)

    # Variational compression
    overlap = 0 
    overlap_before = 1
     
    println("Compressing down to: $Dcut") 
    
    for sweep ∈ 1:max_sweeps            
        _left_sweep_var!!(ϕ, env, ψ, Dcut)
        overlap = _right_sweep_var!!(ϕ, env, ψ, Dcut)

        diff = abs(overlap_before - abs(overlap))
        println("Convergence: ", diff)

        if diff < tol
            println("Finished in $sweep sweeps (of $max_sweeps).")
            break
        else
            overlap_before = overlap
        end    
    end
    return ϕ  
end

function _left_sweep_var!!(ϕ::MPS, env::Vector{<:AbstractMatrix}, ψ::MPS, Dcut::Int)
    S = eltype(ϕ)
    
    # overwrite the overlap
    env[end] = ones(S, 1, 1)

    for i ∈ length(ψ):-1:1
        L = env[i]
        R = env[i+1]

        # optimize site
        M = ψ[i]
        @tensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β) 
        
        # right canonize it
        @cast MM[x, (σ, y)] |= M̃[x, σ, y]
        Q = rq(MM, Dcut)

        d = size(M, 2)
        @cast B[x, σ, y] |= Q[x, (σ, y)] (σ:d)

        # update ϕ and right environment 
        ϕ[i] = B
        A = ψ[i]

        @tensor RR[x, y] := A[x, σ, α] * R[α, β] * conj(B[y, σ, β]) order = (β, α, σ)
        env[i] = RR
    end    
end

function _right_sweep_var!!(ϕ::MPS, env::Vector{<:AbstractMatrix}, ψ::MPS, Dcut::Int)
    S = eltype(ϕ)
    
    # overwrite the overlap
    env[1] = ones(S, 1, 1)

    for i ∈ 1:length(ψ)
        L = env[i]
        R = env[i+1]

        # optimize site
        M = ψ[i]
        @tensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β) 
                   
        # left canonize it
        @cast B[(x, σ), y] |= M̃[x, σ, y]
        Q = qr(B, Dcut)

        d = size(ϕ[i], 2)
        @cast A[x, σ, y] |= Q[(x, σ), y] (σ:d)

        # update ϕ and left environment 
        ϕ[i] = A
        B = ψ[i]

        @tensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end
    return real(env[end][1])
end

