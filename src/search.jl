
function MPO(ig::MetaGraph, nbrs::Vector{Int}, σ::Int, β::T=1) where T <: Number

    L = nv(ig)
    O = MPO{AbstractArray{T}, 4}(L)

    C = I(2)
    δ = I(2)

    for n ∈ nbrs
        J = get_prop(ig, σ, n, :J)
        B = [ exp(-β * x * J * y) for x ∈ [-1, 1], y ∈ [-1, 1] ]

        @cast W[x, σ, y, η] := B[σ, x] * δ[σ, η] * δ[x, y]
    end   
end

function MPS(ig::MetaGraph, β::Number=1, Dcut::Number, tol::Number, max_sweeps::Int) 

    for σ ∈ vertices(ig)
        
        nbrs = neighbors(ig, σ)
        O = MPO(nbrs, σ, β)
        dot!(O, r)

        D = _bondDim(r)
        if D > Dcut
           ρ = compress(r, Dcut, tol, max_sweeps)
        end    
    end    
    return ρ     
end    

function low_energy_spectrum(J::MetaGraph, β::Number=1, Dcut::Number)
    marginal_prob = Dict()
    
    #ρ = MPS()
    #canonize!(ρ, :right)
end
