export truncate!, canonise!

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

"""
function compress(ψ::MPS{T}, Dcut::Int, tol::Number, max_sweeps::Int=4) where {T}
    S = eltype(ψ)
    size = length(ϕ) 

    # Initial guess - truncated ψ 
    ϕ = copy(ψ)
    truncate!(ϕ, :right, Dcut)

    # Create left environment 
    L = left_env(ϕ, ψ)
    overlap = L[end][1]

    # Initialize right environment 
    R = Vector{T}(undef, size)

    while diff > tol 

        if sweep > max_sweep
            println("Max number of sweeps (sweep) reached.")
        else    
            for i ∈ 1:
            # Optimize
        end    
    end
    return ϕ    
end
"""
