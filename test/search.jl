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

    @testset "Gibbs pure state (MPS)" begin
        d = 2
        L = nv(ig)
        dims = fill(d, L)

        @info "Generating exact Gibbs state - |ρ>" d L dims β ϵ

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

        @info "Generating MPS from exact |ρ>"
        rψ = MPS(ψ)
        @test dot(rψ, rψ) ≈ 1

        lψ = MPS(ψ, :left)
        @test dot(lψ, lψ) ≈ 1
 
        vlψ = tensor(lψ)
        vrψ = tensor(rψ)

        vψ = vec(ψ)
        vψ /= norm(vψ)

        @test abs(1 - abs( dot( vec(vlψ), vec(vrψ)) ) ) < ϵ
        @test abs(1 - abs( dot( vec(vlψ), vψ) ) ) < ϵ

        @info "Generating MPS from gates"

        #=
        Gψ = MPS(ig, mps_param, gibbs_param) 

        @test bond_dimension(Gρ) > 1
        @test dot(Gρ, Gρ) ≈ 1
        @test_nowarn verify_bonds(ρ)

        @test abs(1 - abs(dot(Gψ, rψ) ) ) < ϵ # this does not work
        =#

        @info "Verifying probabilities" L β

        for σ ∈ states
            p = dot(rψ, σ) 
            r = dot(rψ, proj(σ, dims), rψ)

            @test p ≈ r 
            @test ϱ[idx.(σ)...] ≈ p
        end 
    end


    #=
    @info "Verifying low energy spectrum"
    states, probab, pCut = _spectrum(ρ, max_states)
    @info "The largest discarded probability" pCut
    @test energy.(states, Ref(ig)) ≈ energies
    @test states == states_bf
    =#
    
end