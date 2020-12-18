using MetaGraphs
using LightGraphs
using GraphPlot

#L = 4
L = 2
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"  

ig = ising_graph(instance, N)
set_prop!(ig, :β, 1.) #rand(Float64))
#r = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
r = [2, 2, 2, 2]
set_prop!(ig, :rank, r)
set_prop!(ig, :dβ, 0.01)

ϵ = 1E-8
D = 16
var_ϵ = 1E-8
sweeps = 30
#type = "log"
type = "lin"
β = [get_prop(ig, :β)]
dβ = [get_prop(ig, :dβ)]
control = MPSControl(D, var_ϵ, sweeps, β, dβ, type) 
states = all_states(get_prop(ig, :rank))
ϱ = gibbs_tensor(ig)
@test sum(ϱ) ≈ 1

@testset "Verifying gate operations" begin
    rank = get_prop(ig, :rank)
    @info "Testing MPS2"

    rψ2 = MPS2(ig, control)
    for max_states ∈ [1, N, 2*N, N^2]
        @info "Testing spectrum_new"
        states_new, prob_new, pCut_new = spectrum_new(rψ2, max_states)            

        eng_new = zeros(length(prob_new))
        for (j, p) ∈ enumerate(prob_new)
            σ = states_new[j, :]
            eng_new[j] = energy(σ, ig)
        end
            
        perm = partialsortperm(eng_new, 1:max_states)
        eng_new = eng_new[perm]
        states_new = states_new[perm, :]
        prob_new = prob_new[perm]
        state = states_new[1, :]
        @info "The largest discarded probability" pCut_new
        @info "State with the lowest energy" state
        @info "Probability of the state with the lowest energy" prob_new[1]
        @info "The lowest energy" eng_new[1]
    end
end
