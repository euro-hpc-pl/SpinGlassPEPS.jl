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
    β_schedule = [dβ, dβ, dβ, dβ]

    mps = MPS_control(Dcut, var_tol, max_sweeps) 
    gibbs = Gibbs_control(β, β_schedule)

    ρ = MPS_from_gates(ig, mps, gibbs) 

    show(ρ)
    @test _verify_bonds(ρ)

    canonise!(ρ)
    @test dot(ρ, ρ) ≈ 1
end