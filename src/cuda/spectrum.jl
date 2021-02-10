# needs testing!!

function CuMPS(ig::MetaGraph, mps::MPSControl)
    L = nv(ig)

    # control for MPS
    Dcut = mps.max_bond
    tol = mps.var_ϵ
    max_sweeps = mps.max_sweeps

    # control for Gibbs state
    β = gibbs.β
    schedule = gibbs.β_schedule

    @assert β ≈ sum(schedule) "Incorrect β schedule."

    # prepare ~ Hadamard state as MPS
    prod_state = fill([1., 1.], nv(ig))
    ρ = MPS(prod_state)

    for dβ ∈ schedule, i ∈ 1:L
        _apply_bias!(ρ, ig, dβ, i) 

        nbrs = unique_neighbors(ig, i)
        if !isempty(nbrs)
            for j ∈ nbrs 
                _apply_exponent!(ρ, ig, dβ, i, j) 
            end

            _apply_projector!(ρ, i)

            for l ∈ setdiff(1:L, union(i, nbrs)) 
                _apply_nothing!(ρ, l) 
            end
        end

        # reduce bond dimension
        if bond_dimension(ρ) > Dcut
            ρ = compress(ρ, Dcut, tol, max_sweeps) 
        end
    end
    ρ
end