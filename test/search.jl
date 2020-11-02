using MetaGraphs
using LightGraphs
using GraphPlot

L = 4
N = L^2

instance = "./instances/$(N)_001.txt"  
#instance = "./instances/lattice_$L.txt" 

ig = ising_graph(instance, N)

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
        ρ = MPS_from_gates(ig, mps_param, gibbs_param) 

        show(ρ)
        @test_nowarn _verify_bonds(ρ)

        canonise!(ρ, :right)
        @test dot(ρ, ρ) ≈ 1




        
        max_states = 4
        states, probab, pCut = _spectrum(ρ, max_states)
        states_bf, energies = _brute_force(ig, max_states)
        
        @info "The largest discarded probability" pCut
        @test energy.(states, Ref(ig)) ≈ energies
        #@test states == states_bf
    end

    #@testset "Generate Gibbs state exactly" begin
    #    r = gibbs_tensor(ig, gibbs_param)
    #    @test sum(r) ≈ 1
    #end
end