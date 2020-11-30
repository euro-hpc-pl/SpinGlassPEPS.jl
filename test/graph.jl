using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

M = 2
N = 2
T = 4

C = 2 * N * M * T

instance = "$(@__DIR__)/instances/chimera_droplets/$(N)power/001.txt"  

@testset "Chimera graph" begin

    ig = ising_graph(instance, N)
    chimera = Chimera((M, N, T), ig)

    #=
    @info "Decomposing Chimera graph into unit cells." M, N, T, C

    unit_cells = []
    for i ∈ 1:N
        for j ∈ 1:M
            cell = chimera_cell(chimera, i, j)
            @test nv(cell) == 2 * T
            push!(unit_cells, cell)
        end
    end

    @info "Putting Chimera back together." 

    chimera_new = zero(chimera)
    for g ∈ unit_cells
        chimera_new = union(chimera_new, g)
    end

    @test chimera == chimera_new
    =#
end
