using MetaGraphs
using LightGraphs
using GraphPlot

L = 4
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"  

ig = ising_graph(instance, N)
set_prop!(ig, :β, 1.) #rand(Float64))
r = fill(2, N)
set_prop!(ig, :rank, r)
set_prop!(ig, :dβ, 0.001)

ϵ = 1E-5
D = 16
var_ϵ = 1E-8
sweeps = 40
type1 = :log
type2 = :lin
β = [get_prop(ig, :β)]
dβ = [get_prop(ig, :dβ)] 
control = MPSControl(D, var_ϵ, sweeps, β, dβ) 

states = all_states(get_prop(ig, :rank))
ϱ = gibbs_tensor(ig)
@test sum(ϱ) ≈ 1

@testset "Verifying gate operations" begin
    rank = get_prop(ig, :rank)
    @info "Testing MPS"

    rψ1 = MPS(ig, control, type1)
    rψ2 = MPS(ig, control, type2)
    rψ3 = MPS(ig, control)
    overlap12 = dot(rψ1, rψ2)
    @test overlap12 ≈ 0.9999998
    overlap13 = dot(rψ1, rψ3)
    @test overlap13 ≈ 1 
    overlap23 = dot(rψ2, rψ3)
    @test overlap23 ≈ 0.9999998

    for max_states ∈ [1, N, 2*N, N^2]
        @info "Testing spectrum_new"
        states_new1, prob_new1, pCut_new1 = solve_new(rψ1, max_states)   
        states_new2, prob_new2, pCut_new2 = solve_new(rψ2, max_states)
        states_new3, prob_new3, pCut_new3 = solve_new(rψ3, max_states)         

        eng_new1 = zeros(length(prob_new1))
        eng_new2 = zeros(length(prob_new2))
        eng_new3 = zeros(length(prob_new3))
        for (j, p) ∈ enumerate(prob_new1)
            σ = states_new1[j, :]
            eng_new1[j] = energy(σ, ig)
        end
        for (j, p) ∈ enumerate(prob_new2)
            σ = states_new2[j, :]
            eng_new2[j] = energy(σ, ig)
        end
        for (j, p) ∈ enumerate(prob_new3)
            σ = states_new3[j, :]
            eng_new3[j] = energy(σ, ig)
        end
            
        perm1 = partialsortperm(eng_new1, 1:max_states)
        eng_new1 = eng_new1[perm1]
        states_new1 = states_new1[perm1, :]
        prob_new1 = prob_new1[perm1]
        state1 = states_new1[1, :]
        @info "Testing MPS2 - logarithmic"
        @info "The largest discarded probability" pCut_new1
        @info "State with the lowest energy" state1
        @info "Probability of the state with the lowest energy" prob_new1[1]
        @info "The lowest energy" eng_new1[1]

        perm2 = partialsortperm(eng_new2, 1:max_states)
        eng_new2 = eng_new2[perm2]
        states_new2 = states_new2[perm2, :]
        prob_new2 = prob_new2[perm2]
        state2 = states_new2[1, :]
        @info "Testing MPS2 - linear"
        @info "The largest discarded probability" pCut_new2
        @info "State with the lowest energy" state2
        @info "Probability of the state with the lowest energy" prob_new2[1]
        @info "The lowest energy" eng_new2[1]

        perm3 = partialsortperm(eng_new3, 1:max_states)
        eng_new3 = eng_new3[perm3]
        states_new3 = states_new3[perm3, :]
        prob_new3 = prob_new3[perm3]
        state3 = states_new3[1, :]
        @info "Testing MPS"
        @info "The largest discarded probability" pCut_new3
        @info "State with the lowest energy" state3
        @info "Probability of the state with the lowest energy" prob_new3[1]
        @info "The lowest energy" eng_new3[1]

    end
end