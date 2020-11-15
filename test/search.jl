using MetaGraphs
using LightGraphs
using GraphPlot

L = 2
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"  
ig = ising_graph(instance, N)

@testset "MPS from gates" begin

    ϵ = 1E-14

    Dcut = 6
    var_tol = 1E-8
    max_sweeps = 4

    β = 1
    dβ = 0.25
    β_schedule = [β] #[dβ for _ ∈ 1:4]

    gibbs_param = GibbsControl(β, β_schedule)
    mps_param = MPSControl(Dcut, var_tol, max_sweeps) 

    @testset "Exact Gibbs pure state (MPS)" begin
        d = 2
        L = nv(ig)
        dims = fill(d, L)

        @info "Generating Gibbs state - |ρ>" d L dims β ϵ

        ϱ = gibbs_tensor(ig, gibbs_param)
        @test sum(ϱ) ≈ 1

        states = all_states(dims) 
        ψ = ones(dims...)

        for σ ∈ states
            for i ∈ 1:L
                h = get_prop(ig, i, :h)

                nbrs = unique_neighbors(ig, i)
                ψ[idx.(σ)...] *= exp(0.5 * β * h * σ[i]) 

                for j ∈ nbrs
                    J = get_prop(ig, i, j, :J)
                    ψ[idx.(σ)...] *= exp(0.5 * β * σ[i] * J * σ[j]) 
                end      
            end     
        end

        ρ = abs.(ψ) .^ 2 
        @test ρ / sum(ρ) ≈ ϱ

        @info "Generating MPS from |ρ>"
        rψ = MPS(ψ)
        @test dot(rψ, rψ) ≈ 1

        lψ = MPS(ψ, :left)
        @test dot(lψ, lψ) ≈ 1
 
        vlψ = vec(tensor(lψ))
        vrψ = vec(tensor(rψ))

        vψ = vec(ψ)
        vψ /= norm(vψ)

        @test abs(1 - abs(dot(vlψ, vrψ))) < ϵ
        @test abs(1 - abs(dot(vlψ, vψ))) < ϵ

        @info "Verifying probabilities" L β

        for σ ∈ states
            p = dot(rψ, σ) 
            r = dot(rψ, proj(σ, dims), rψ)

            @test p ≈ r 
            @test ϱ[idx.(σ)...] ≈ p
        end 

        max_states = N^2
        @info "Verifying low energy spectrum" max_states

        @test_nowarn is_right_normalized(rψ)

        states, prob = spectrum(rψ, max_states)
        states_bf, energies = brute_force(ig, max_states)

        #@info "The largest discarded probability" pCut

        for (j, (p, e)) ∈ enumerate(zip(prob, energies))
            σ = states[:, j]
            @test ϱ[idx.(σ)...] ≈ p
            @test abs(energy(σ, ig) - e) < ϵ
        end
    end
    
    #=
    @info "Generating MPS from gates"
    Gψ = MPS(ig, mps_param, gibbs_param) 

    @test bond_dimension(Gρ) > 1
    @test dot(Gρ, Gρ) ≈ 1
    @test_nowarn verify_bonds(ρ)

    @test abs(1 - abs(dot(Gψ, rψ) ) ) < ϵ # this does not work
    =#
end