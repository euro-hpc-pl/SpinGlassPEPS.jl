using MetaGraphs
using LightGraphs
using GraphPlot

L = 2
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"  

ig = ising_graph(instance, N)
verify_ρ = false
verify_spectrum = false

@testset "MPS from gates" begin

    Dcut = 6
    var_tol = 1E-8
    max_sweeps = 4

    β = 1
    dβ = 0.25
    β_schedule = [β] #[dβ for _ ∈ 1:4]

    gibbs_param = GibbsControl(β, β_schedule)
    mps_param = MPSControl(Dcut, var_tol, max_sweeps) 

    ϱ = gibbs_tensor(ig, gibbs_param)

    @testset "Gibbs pure state (MPS)" begin
        L = nv(ig)

        states = ising.(digits.(0:2^L-1, base=2, pad=L))
        ψ = ones(fill(2, L)...)

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

        #@test MPS(ψ) ≈ MPS(ig, mps_param, gibbs_param) 
    end

#=
    @testset "Local gates (bias)" begin
        β = rand()
        L = nv(ig)

        ρ = HadamardMPS(L)
        @test dot(ρ, ρ) ≈ 1

        bias = [get_prop(ig, i, :h) for i ∈ 1:L]
        ϕ = [[exp(-0.5 * β * h), exp(0.5 * β * h)] / sqrt(2) for h ∈ bias]
        ϱ = MPS(ϕ)

        for i ∈ 1:L
            _apply_bias!(ρ, ig, β, i)
        end     

        @test bond_dimension(ρ) == 1
        @test dot(ρ, ρ) ≈ prod([cosh(β * h) for h ∈ bias])
    end
 
    if false
    @testset "Low energy spectrum from ρ" begin
        ρ = MPS(ig, mps_param, gibbs_param) 
        show(ρ)

        @test bond_dimension(ρ) > 1
        @test dot(ρ, ρ) ≈ 1
        @test_nowarn verify_bonds(ρ)

        max_states = 4
        @assert max_states <= N
        states_bf, energies = brute_force(ig, max_states)

        if verify_ρ
            @info "Verifying ρ MPS"
            rho = gibbs_tensor(ig, gibbs_param)

            if N <= 4 display(rho) end

            for (i, σ) ∈ enumerate(states_bf)
                p = dot(ρ, σ)  
                @info "probability for a given config" i p

                @test p in rho
                @test p ≈ dot(ρ, proj(σ), ρ)                
                @test 0 <= p <= 1
    
                #@test rho[idx.(σ)...] ≈ p
            end   
        end
    end
        if verify_spectrum

            @info "Verifying spectrum"
            states, probab, pCut = _spectrum(ρ, max_states)
            @info "The largest discarded probability" pCut
            @test energy.(states, Ref(ig)) ≈ energies
            #@test states == states_bf
        end 
    
    end
    =#
end