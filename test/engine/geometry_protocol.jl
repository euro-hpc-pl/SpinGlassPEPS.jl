# Conformance tests for the geometry protocol (see src/geometry.jl).
#
# Every geometry x layout combination the solver supports must implement the
# full interface; this catches missing methods at test time instead of deep
# inside a solve. Extend SUPPORTED when adding a geometry or layout.

using SpinGlassPEPS.SpinGlassEngine
using SpinGlassPEPS.SpinGlassEngine: tensor_map, gauges_list, nodes_search_order_Mps
using SpinGlassPEPS.SpinGlassEngine: conditional_probability, projectors_site_tensor
using SpinGlassPEPS.SpinGlassEngine: boundary, update_energy
using Test

SUPPORTED = Dict(
    SquareSingleNode => (EnergyGauges, GaugesEnergy, EngGaugesEng),
    KingSingleNode => (EnergyGauges, GaugesEnergy, EngGaugesEng),
    SquareDoubleNode => (EnergyGauges, GaugesEnergy),
    SquareCrossDoubleNode => (EnergyGauges, GaugesEnergy),
)

@testset "Geometry protocol conformance" begin
    for (G, layouts) ∈ SUPPORTED, L ∈ layouts
        GL = G{L}
        @testset "$(GL)" begin
            @test hasmethod(G, Tuple{Int,Int})
            @test hasmethod(tensor_map, Tuple{Type{GL},Type{Dense},Int,Int}) ||
                  hasmethod(tensor_map, Tuple{Type{G},Type{Dense},Int,Int})
            @test hasmethod(tensor_map, Tuple{Type{GL},Type{Sparse},Int,Int}) ||
                  hasmethod(tensor_map, Tuple{Type{G},Type{Sparse},Int,Int})
            @test hasmethod(gauges_list, Tuple{Type{GL},Int,Int})
            @test hasmethod(MpoLayers, Tuple{Type{GL},Int})
            @test hasmethod(conditional_probability, Tuple{Type{GL},MpsContractor,Vector{Int}})
            @test hasmethod(nodes_search_order_Mps, Tuple{PEPSNetwork{GL,Dense}})
            @test hasmethod(boundary, Tuple{Type{GL},MpsContractor,Node})
            # constructible without a network: layers and gauge tables
            ml = MpoLayers(GL, 4)
            @test ml isa MpoLayers
            @test !isempty(ml.main) && !isempty(ml.dress) && !isempty(ml.right)
            gl = gauges_list(GL, 4, 4)
            @test gl isa Vector{SpinGlassEngine.GaugeInfo}
        end
    end
end
