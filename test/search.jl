using MetaGraphs
using LightGraphs
using GraphPlot

L = 3
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"  
#instance = "$(@__DIR__)/instances/lattice_$L.txt" 

ig = ising_graph(instance, N)
verify_ρ = false
verify_spectrum = false

@testset "MPS from gates" begin

    Dcut = 16
    var_tol=1E-8
    max_sweeps = 4

    β = 2
    dβ = 0.5
    β_schedule = [dβ for _ ∈ 1:4]

    gibbs_param = GibbsControl(β, β_schedule)
    mps_param = MPSControl(Dcut, var_tol, max_sweeps) 

    @testset "Low energy spectrum from ρ" begin
        ρ = MPS(ig, mps_param, gibbs_param) 

        show(ρ)
        @test_nowarn SpinGlassPEPS._verify_bonds(ρ)

        max_states = 1
        states_bf, energies = _brute_force(ig, max_states)

        if verify_ρ
            @info "Verifying ρ MPS"

            #all_states = _toIsing.(digits.(0:2^N-1, base=2, pad=N))
            #states = all_states[1:10]
            states = states_bf

            r = gibbs_tensor(ig, gibbs_param)

            for (i, σ) ∈ enumerate(states)
                p = dot(ρ, σ)
                @info "pdo" i p
                @test p ≈ dot(ρ, _projector(σ), ρ)
                
                @test p <= 1.
                @test p >= 0.
    
                #@test r[_toIdx.(σ)...] ≈ p
            end   
        end

        if verify_spectrum
            canonise!(ρ, :right)
            @test dot(ρ, ρ) ≈ 1

            @info "Verifying spectrum"
            states, probab, pCut = _spectrum(ρ, max_states)
            @info "The largest discarded probability" pCut
            @test energy.(states, Ref(ig)) ≈ energies
            #@test states == states_bf
        end 
    end

    #@testset "Generate Gibbs state exactly" begin
    #    r = gibbs_tensor(ig, gibbs_param)
    #    @test sum(r) ≈ 1
    #end
end