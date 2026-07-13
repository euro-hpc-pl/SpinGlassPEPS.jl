@testset "Graph to qubo" begin
    graph = [-3 2 2; 0 -3 2; 0 0 -3 ]
    qubo = graph_to_qubo(graph)
    @test qubo == [-14 8 8; 0 -14 8; 0 0 -14 ]
end 