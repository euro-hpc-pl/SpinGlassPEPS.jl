import SpinGlassPEPS: last_m_els, M2graph
import SpinGlassPEPS: spins2binary, binary2spins
include("test_helpers.jl")


@testset "helpers" begin
    M = ones(4,4)
    fullM2grid!(M, (2,2))
    @test M == [1.0 1.0 1.0 0.0; 1.0 1.0 0.0 1.0; 1.0 0.0 1.0 1.0; 0.0 1.0 1.0 1.0]
    @test spins2binary([1,1,-1]) == [1,1,0]
    @test binary2spins([0,0,1]) == [-1,-1,1]
end

@testset "graph representation" begin
    M = ones(2,2)

    # graph for mps
    g = M2graph(M)
    @test collect(vertices(g)) == [1,2]

    @test props(g, 1)[:h] == 1.
    @test props(g, 2)[:h] == 1.
    @test props(g, 1,2)[:J] == 2.
end

@testset "operations on spins" begin
    @test last_m_els([1,2,3,4], 2) == [3,4]
    @test last_m_els([1,2,3,4], 5) == [1,2,3,4]
end
