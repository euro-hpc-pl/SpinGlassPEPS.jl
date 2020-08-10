include("peps.jl")

using Test
using LinearAlgebra

@testset "tensor operations tests" begin
    A = ones(2,2,2,2)
    @test sum_over_last(A) == 2*ones(2,2,2)
    @test set_last(A, -1) == ones(2,2,2)

    A = 0.1*ones(2,2,2)
    C = [1. 2. ; 3. 4.]

    T = contract_ts1(A, C, 2,1)
    @test T[1,:,:] ≈ [0.4 0.6; 0.4 0.6]
end

@testset "adding qubo to  graph" begin
    function make_qubo_t()
        qubo = [(1,1) 0.5; (1,2) 0.5; (1,6) 0.5; (2,2) 0.5; (2,3) 0.5; (2,5) 0.5; (3,3) 0.5; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.5; (4,5) 0.5; (4,9) 0.5; (5,5) 0.5; (5,6) 0.5; (5,8) 0.5; (6,6) 0.5; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.5; (7,8) 0.5; (8,8) 0.5; (8,9) 0.5; (9,9) 0.5])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    mg_t = make_graph3x3()
    qubo = make_qubo_t()
    add_qubo2graph(mg_t, qubo)

    @test collect(vertices(mg_t)) == (1:9)

    @test props(mg_t, 1)[:h] == 0.5
    @test props(mg_t, 2)[:h] == 0.5
    @test props(mg_t, 3)[:h] == 0.5
    @test props(mg_t, 4)[:h] == 0.5
    @test props(mg_t, Edge(1,6))[:J] == 0.5

    @test props(mg_t, Edge(1,2))[:type] == ["r", "l"]
    @test props(mg_t, Edge(1,6))[:type] == ["d", "u"]
    @test props(mg_t, Edge(6,1))[:type] == ["d", "u"]
end

@testset "testing of tensor generator" begin
    β = 1.

    @test delta(0,-1) == 1
    @test delta(-1,1) == 0
    @test delta(1,1) == 1
    @test c(0, 1., 20) == 1
    @test Tgen(0,0,0,0,-1,0.,0.,1.) == exp(β)
end


@testset "axiliary functions for tensor creation" begin

    @test sort2lrud(["d", "l"]) == ["l", "d"]
    @test sort2lrud(["u", "d", "l", "r"]) == ["l", "r", "u", "d"]
    @test index2physical(2) == 1
    @test index2physical(1) == -1
    @test_throws ErrorException index2physical(-1)
end

@testset "tensor on graph testing" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    mg = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph(mg, qubo);

    @test bond_dirs(mg, 1)  == ["r", "d"]
    @test bond_dirs(mg, 5) == ["l", "r", "u", "d"]
    @test bond_dirs(mg, 9) == ["l", "u"]

    @test get_modes(mg, 1) == [2, 4]
    @test get_modes(mg, 5) == [1, 2, 3, 4]
    @test get_modes(mg, 9) == [1, 3]

    @test getJs(mg, 1) == (0.5, 0.5, 0.2)
    T = makeTensor(mg, 1)

    @test T[1] == 0.4493289641172216

    T1 = [Tgen(l,r,u,d,s,0.5, 0.5, 0.2)  for s in [-1, 1] for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    makeTensor(mg, 5) == reshape(T1, (2,2,2,2,2))

    T2 = [Tgen(0,r,0,d,s,0.5, 0.5, 0.2) for s in [-1, 1] for d in [-1, 1] for r in [-1, 1]]
    makeTensor(mg, 1) == reshape(T2, (2,2,2))

    T3 = [Tgen(l,r,u,d,1,0.5, 0.5, 0.2)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    add_tensor2vertex(mg, 5, 1)
    @test props(mg, 5)[:tensor] == reshape(T3, (2,2,2,2))

    T4 = [Tgen(l,r,u,d,-1,0.5, 0.5, 0.2)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    T5 = T3+T4
    add_tensor2vertex(mg, 5)
    @test props(mg, 5)[:tensor] == reshape(T5, (2,2,2,2))
end


@testset "filling grid with tensors" begin

    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    mg = make_graph3x3();
    mg1 = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph(mg, qubo);
    add_qubo2graph(mg1, qubo);

    @test props(mg, Edge(4,5))[:modes] == [0,0]

    @test props(mg, Edge(4,5))[:type] == ["l", "r"]
    @test props(mg, Edge(5,2))[:type] == ["d", "u"]
    @test props(mg, Edge(1,2))[:type] == ["r", "l"]

    @test_throws ErrorException add_tensor2vertex(mg, 1, 3)

    for i in 1:9
        add_tensor2vertex(mg, i)
    end

    s = [1,1,1,-1,-1,-1,1,1,1]
    for i in 1:9
        add_tensor2vertex(mg1, i, s[i])
    end

    @test props(mg, Edge(4,5))[:modes] == [1,2]
    @test props(mg, Edge(1,2))[:modes] == [1,1]
    @test props(mg, Edge(2,5))[:modes] == [3,3]
    @test props(mg, Edge(2,3))[:modes] == [2,1]
    @test props(mg, Edge(3,4))[:modes] == [2,2]

    @test props(mg1, Edge(4,5))[:modes] == [1,2]

    @test norm(props(mg, 5)[:tensor] - props(mg1, 5)[:tensor]) > 2.
end

@testset "contract vertices" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    mg = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph(mg, qubo);

    for i in 1:9
        add_tensor2vertex(mg, i)
    end


    cc = contract_vertices(mg, 5,8)
    cc = contract_vertices(mg, 6,7)
    cc = contract_vertices(mg, 4,9)
    T = props(mg, 5)[:tensor]
    @test T[1,1,1,1,1] == 0.33287108369807955
    @test T[1,2,1,2,1] == 4.481689070338066
    @test ndims(T) == 5
    @test ndims(props(mg, 4)[:tensor]) == 3
    @test props(mg, Edge(5,6))[:modes] == [1,1,4,3]
    @test props(mg, Edge(5,4))[:modes] == [1,2,3,5]
    @test props(mg, Edge(5,2))[:modes] == [3,3]

end
