export truncate!, canonise!

function canonise!(ψ::MPS{T}) where T <: AbstractArray{<:Number, 3}
    canonise!(ψ, "left")
    canonise!(ψ, "right")
end 

function canonise!(ψ::MPS{T}, type::String) where T <: AbstractArray{<:Number, 3}
    if type == "right"
        _left_sweep_SVD!(ψ, typemax(Int))
    elseif type == "left"
        _right_sweep_SVD!(ψ, typemax(Int))
    else
        error("Choose: left or right")    
    end    
end

function truncate!(ψ::MPS{T}, type::String, Dcut::Int) where T <: AbstractArray{<:Number, 3}
    if type == "right"
        _left_sweep_SVD!(ψ, Dcut)
    elseif type == "left"
        _right_sweep_SVD!(ψ, Dcut)
    else
        error("Choose: left or right")    
    end    
end

function _right_sweep_SVD!(ψ::MPS{T}, Dcut::Int) where T <: AbstractArray{<:Number, 3}
    Σ = V = ones(eltype(T), 1, 1)

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
        ψ.tensors[i] = B
    end
end

function _left_sweep_SVD!(ψ::MPS{T}, Dcut::Int) where T <: AbstractArray{<:Number, 3}
    Σ = U = ones(eltype(T), 1, 1)

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
        ψ.tensors[i] = A
    end
end