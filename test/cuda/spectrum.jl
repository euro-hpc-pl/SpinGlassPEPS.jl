using MetaGraphs
using LightGraphs

L = 2
N = L^2

instance = "$(@__DIR__)/../instances/$(N)_001.txt"  

ig = ising_graph(instance, N)
set_prop!(ig, :β, 1.)
r = fill(2, N)
set_prop!(ig, :rank, r)
set_prop!(ig, :dβ, 0.01)

sgn = -1.
ϵ = 1E-6
D = prod(r) + 1
var_ϵ = 1E-8
sweeps = 4
schedule = [get_prop(ig, :β)]
dβ = [get_prop(ig, :dβ)]
control = MPSControl(D, var_ϵ, sweeps, schedule, dβ) 

states = all_states(get_prop(ig, :rank))
ϱ = cu(gibbs_tensor(ig))
@test sum(ϱ) ≈ 1

@testset "MPS from gates" begin

    @testset "Exact Gibbs pure state (MPS)" begin
        L = nv(ig)
        β = get_prop(ig, :β)
        rank = get_prop(ig, :rank)

        @info "Generating Gibbs state - |ρ>" L rank β ϵ

        ψ = ones(rank...)

        for σ ∈ states
            for i ∈ 1:L
                h = get_prop(ig, i, :h)

                nbrs = unique_neighbors(ig, i)
                ψ[idx.(σ)...] *= exp(sgn * 0.5 * β * h * σ[i]) 

                for j ∈ nbrs
                    J = get_prop(ig, i, j, :J)
                    ψ[idx.(σ)...] *= exp(sgn * 0.5 * β * σ[i] * J * σ[j]) 
                end      
            end     
        end
        ψ = cu(ψ)
        ρ = abs.(ψ) .^ 2 
        @test ρ / sum(ρ) ≈ ϱ

        @info "Generating MPS from |ρ>"
        rψ = CuMPS(ψ)
        @test dot(rψ, rψ) ≈ 1
        @test_nowarn is_right_normalized(rψ)

        lψ = CuMPS(ψ, :left)
        @test dot(lψ, lψ) ≈ 1
 
        vlψ = vec(tensor(lψ))
        vrψ = vec(tensor(rψ))

        vψ = vec(ψ)
        vψ /= norm(vψ)
        
        @test abs(1 - abs(CUDA.dot(vlψ, vrψ))) < ϵ
        @test abs(1 - abs(CUDA.dot(vlψ, vψ))) < ϵ

        @info "Verifying MPS from gates"

        Gψ = CuMPS(ig, control) 

        @test_nowarn is_right_normalized(Gψ)
        @test bond_dimension(Gψ) > 1
        @test dot(Gψ, Gψ) ≈ 1
        @test_nowarn verify_bonds(Gψ)

        @test abs(1 - abs(dot(Gψ, rψ))) < ϵ 

        @info "Verifying probabilities" L β
        for σ ∈ states
            p = dot(rψ, σ) 
            r = dot(rψ, cuproj(σ, rank), rψ)

            @test p ≈ r 
            @test Array(ϱ)[idx.(σ)...] ≈ p
        end 

        for max_states ∈ [1, N, 2*N, N^2]

            @info "Verifying low energy spectrum" max_states
            @info "Testing spectrum"
            states, prob, pCut = solve(rψ, max_states)
            sp = brute_force(ig, num_states = max_states)

            @info "The largest discarded probability" pCut
      
            for (j, (p, e)) ∈ enumerate(zip(prob, sp.energies))
                σ = states[:, j]
                @test Array(ϱ)[idx.(σ)...] ≈ p
                @test abs(energy(σ, ig) - e) < ϵ
            end

            @info "Testing spectrum_new"
            states_new, prob_new, pCut_new = solve_new(rψ, max_states)            

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
            @test maximum(prob_new) > pCut_new
            @info "State with the lowest energy" state
            @info "Probability of the state with the lowest energy" prob_new[1]
            @info "The lowest energy" eng_new[1]
           
        end

    end
end