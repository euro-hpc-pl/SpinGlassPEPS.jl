include("peps.jl")

using Test

@testset "tensor operations tests" begin
    A = ones(2,2,2,2)
    @test sum_over_last(A) == 2*ones(2,2,2)
    @test set_last(A, -1) == ones(2,2,2)

    A = 0.1*ones(2,2,2)
    modesA = ["u", "d", "r"]
    C = [1. 2. ; 3. 4.]
    modesC = ["u", "r"]
    scheme = ["d", "u"]

    T, v = contract_ts(A, C, modesA, modesC, scheme)
    @test T[1,:,:] ≈ [0.4 0.6; 0.4 0.6]
    @test v == ["u", "r1", "r2"]
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

    @test props(mg_t, 1)[:bonds]["r"] == 2
    @test props(mg_t, 2)[:bonds]["r"] == 3
    @test props(mg_t, 3)[:bonds]["r"] == 0
    @test props(mg_t, 4)[:bonds]["r"] == 0
    @test props(mg_t, 5)[:bonds]["r"] == 4
end

@testset "testing of tensor generator" begin
    β = 1.

    @test delta(0,-1) == 1
    @test delta(-1,1) == 0
    @test delta(1,1) == 1
    @test c(0, 1., 20) == 1
    @test Tgen(0,0,0,0,-1,0.,0.,1.) == exp(β)
end


@testset "tensor type testing" begin

    @test index2physical(2) == 1
    @test index2physical(1) == -1

    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    mg = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph(mg, qubo);

    @test get_modes(mg, 1) == (["r", "d"], [2, 4])
    @test getJs(mg, 1) == (0.5, 0.5, 0.2)
    T,m = makeTensor(mg, 1)

    @test m == ["r", "d", "s"]

    T1 = [Tgen(l,r,u,d,s,0.5, 0.5, 0.2)  for s in [-1, 1] for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    @test TensorOnGraph(mg, 5).A == reshape(T1, (2,2,2,2,2))

    T2 = [Tgen(0,r,0,d,s,0.5, 0.5, 0.2) for s in [-1, 1] for d in [-1, 1] for r in [-1, 1]]
    @test TensorOnGraph(mg, 1).A == reshape(T2, (2,2,2))

    T3 = [Tgen(l,r,u,d,1,0.5, 0.5, 0.2)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    A = TensorOnGraph(mg, 5)
    @test set_physical_dim(A, 1).A == reshape(T3, (2,2,2,2))


    T4 = [Tgen(l,r,u,d,-1,0.5, 0.5, 0.2)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
    T5 = T3+T4
    A = TensorOnGraph(mg, 5)
    @test trace_physical_dim(A).A == reshape(T5, (2,2,2,2))
    @test trace_physical_dim(A).bonds_dirs == ["l","r","u","d"]
end


@testset "filling grid with tensors" begin

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

    @test props(mg, 5)[:tensor].bonds_dirs == ["l", "r", "u", "d"]

end
