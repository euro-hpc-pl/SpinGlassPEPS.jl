using MetaGraphs
using LightGraphs
using GraphPlot
using CSV

@testset "Chimera creation" begin
    m = 6
    n = 7
    t = 4
    g = Chimera(m, n, t)
    @test nv(g) == 2m * n * t
    @test ne(g) == t^2 * m * n + m * (n -1) * t + (m - 1) * n * t
    @test g[m, n, 2, t] == 2m * n * t
    @show g[1, 1]
end

 @testset "Chimera graph" begin
     m = 4
     n = 4
     t = 4

     C = 2 * m * n * t

     instance = "$(@__DIR__)/instances/chimera_droplets/$(C)power/001.txt"  
     ig = ising_graph(instance, C)
     chimera = Chimera((m, n, t), ig)

     for e ∈ edges(chimera)
        get_prop(chimera, e, :J) ≈ get_prop(ig, e, :J) 
     end

     for v ∈ vertices(chimera)
        get_prop(chimera, v, :h) ≈ get_prop(ig, v, :h) 
     end

     #=
     linear = LinearIndices((1:m, 1:n))
     for i ∈ 1:m 
        for j ∈ 1:n
            v = linear[i, j]
            cluster = filter_vertices(chimera, :cluster, v)
            println(cluster)
        end
    end
    #=
end
