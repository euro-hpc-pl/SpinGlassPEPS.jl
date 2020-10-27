using MetaGraphs
using LightGraphs
using GraphPlot

L = 3
instance = "./lattice_$L.txt"    
ig = ising_graph(instance, L^2)

@testset "MPS from gates" begin
    Dcut = 16
    dβ = 0.2
    var_tol=1E-8

    ρ = MPS_from_gates(ig, dβ, Dcut, var_tol) 

    show(ρ)

    @test _verify_bonds(ρ)
end