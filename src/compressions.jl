export truncate!, canonise!, compress

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
 

function compress(ψ::MPS{T}, Dcut::Int, tol::Number, max_sweeps::Int=4) where {T}
    # Initial guess - truncated ψ 
    ϕ = copy(ψ)
    truncate!(ϕ, :right, Dcut)

    # Create environment 
    env = left_env(ϕ, ψ)

    # Variational compression
    sweep = 0
    stuck = finish = false
    dist = dist_befor = Inf
     
    println("Compressing down to: $Dcut")
    
    while !(stuck || finish)             
               _left_sweep_var!(ϕ, ψ, env, Dcut)
        dist = _right_sweep_var!(ϕ, ψ, env, Dcut)

        sweep += 1
        println("sweep:", sweep)
        finish = sweep > max_sweeps 

        println("diff:", abs(dist_befor - dist))
        stuck = abs(dist_befor - dist) < tol

        dist_befor = dist
    end

    println("Finished in $sweep sweeps (of $max_sweeps).")
    println("Distance between wave functions is $dist.")
    println("Stuck:", stuck)

    return ϕ    
end

function _left_sweep_var!(ϕ::MPS, ψ::MPS, env::Vector{T}, Dcut::Int) where T <: AbstractMatrix
    S = eltype(ϕ)
    
    # overwrite the overlap
    env[end] = ones(S, 1, 1)

    for i ∈ length(ψ):-1:1

        # get environment (left: attach center, right: as is)
        R = env[i+1]
        L = env[i]

        # optimize site
        M = ψ[i]
        #println( size(LR), size(M), size(RR) )
        @tensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β) 
        
        # right canonize it
        @cast MM[x, (σ, y)] |= M̃[x, σ, y]
        QR_fact = pqrfact(:c, conj(MM), rank=Dcut)

        d = size(M, 2)
        Q = conj(QR_fact[:Q])'
        C = QR_fact[:R]'
        @cast B[x, σ, y] |= Q[x, (σ, y)] (σ:d)

        # update ϕ and right environment 
        ϕ[i] = B
        A = ψ[i]

        @tensor r[x, y] := A[x, σ, α] * R[α, β] * conj(B[y, σ, β]) order = (β, α, σ)
        env[i] = r
    end    
end

function _right_sweep_var!(ϕ::MPS, ψ::MPS, env::Vector{T}, Dcut::Int) where T <: AbstractMatrix
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
        QR_fact = pqrfact(B, rank=Dcut)

        d = size(ϕ[i], 2)
        Q = QR_fact[:Q]
        @cast A[x, σ, y] |= Q[(x, σ), y] (σ:d)

        # update ϕ and left environment 
        ϕ[i] = A
        B = ψ[i]

        @tensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end

    # Calculate ||ψ-ϕ|| assuming ||ψ|| = ||ϕ|| = 1
    return sqrt( 2 * (1 - real(env[end][1]) ) )
end
