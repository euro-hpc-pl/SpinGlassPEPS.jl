using MetaGraphs
using LightGraphs
using GraphPlot

L = 3
N = L^2
instance = "./lattice_$L.txt"    
ig = ising_graph(instance, N)

@testset "MPS from gates" begin

    Dcut = 16
    var_tol=1E-8
    max_sweeps = 4

    β = 1
    dβ = 0.25
    β_schedule = [dβ for _ ∈ 1:4]

    gibbs_param = GibbsControl(β, β_schedule)
    mps_param = MPSControl(Dcut, var_tol, max_sweeps) 

    @testset "Generate ρ" begin
        ρ = MPS(ig, mps_param, gibbs_param) 

        show(ρ)
        @test_nowarn _verify_bonds(ρ)

        canonise!(ρ)
        @test dot(ρ, ρ) ≈ 1
    end

    @testset "Generate Gibbs state exactly" begin
        r = gibbs_tensor(ig, gibbs_param)
        @test sum(r) ≈ 1
    end
end