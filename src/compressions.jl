export truncate!, canonise!

function canonise!(ψ::MPS)
    canonise!(ψ, :left)
    canonise!(ψ, :right)
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

        B = ψ[i]
        C = Diagonal(Σ) * V'

        # attach
        @tensor M[x, σ, y] := C[x, α] * B[α, σ, y]
        @cast   M̃[(x, σ), y] |= M[x, σ, y]
        
        # decompose
        U, Σ, V = psvd(M̃, rank=Dcut)

        # create new    
        d = size(ψ[i], 2)
        @cast B[x, σ, y] |= U[(x, σ), y] (σ:d)
        ψ[i] = B
    end
end

function _left_sweep_SVD!(ψ::MPS, Dcut::Int=typemax(Int))
    Σ = U = ones(eltype(ψ), 1, 1)

    for i ∈ length(ψ):-1:1

        A = ψ[i]
        C = U * Diagonal(Σ)

        # attach
        @tensor M[x, σ, y]   := A[x, σ, α] * C[α, y]
        @cast   M̃[x, (σ, y)] |= M[x, σ, y]

        # decompose
        U, Σ, V = psvd(M̃, rank=Dcut)

        # create new 
        d = size(ψ[i], 2)
        @cast A[x, σ, y] |= V'[x, (σ, y)] (σ:d)
        ψ[i] = A
    end
end