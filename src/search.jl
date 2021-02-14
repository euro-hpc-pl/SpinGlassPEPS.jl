#=
# This is the most general (still semi-sudo-code) of the search function.
# 
export search

function merge(model, partial_sol, partial_eng)
    for (i, s) ∈ enumerate(partial_sol)
        boundary[i] = boundary(model, s)
    end
    idx = partition_into_unique(boundary, partial_eng)
    idx
end

function search(model, k)
    partial_sol = []
    partial_energy = []
    marginal_prob = []
    
    for v ∈ 1:model.size 
        for (i, p) ∈ enumerate(partial_sol)
            cond_prob, new_sol[i] = conditional_probability(model, p)
            new_prob[i] = marginal_prob[i] * cond_prob
            new_energy[i] = partial_energy[i] + energy_difference(model, p)
        end
        new_prob = vec(new_prob)
        new_sol = vec(new_sol)

        idx = merge(model, new_sol, new_energy)
        new_prob = new_prob[idx]
        new_eng = new_eng[idx]
        new_sol = new_sol[idx]

        partialsortperm!(perm, vec(new_prob), 1:k, rev=true)
        marginal_prob = vec(new_prob)[perm]
        partial_sol = reshape(new_sol, size(marginal_prob), v)[perm]
        partial_eng = vec(new_energy)[perm]
        lpCut < last(marginal_prob) ? lpCut = last(marginal_prob) : ()
    end
    partialsortperm!(perm, vec(partial_eng), 1:size(partial_eng), rev=true)
    prob = vec(marginal_prob)[perm]
    sol = reshape(partial_sol, size(marginal_prob), v)[perm]
    eng = vec(partial_energy)[perm]
    
    eng, sol, prob, lpCut
end
=#