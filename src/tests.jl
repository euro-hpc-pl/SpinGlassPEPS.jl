include("notation.jl")
include("alternative_approach.jl")
include("peps.jl")

using Test
using LinearAlgebra

if true
@testset "tensor operations tests" begin
    A = ones(2,2,2,2)
    @test sum_over_last(A) == 2*ones(2,2,2)
    @test set_last(A, -1) == ones(2,2,2)

    A = 0.1*ones(2,2,2)
    C = [1. 2. ; 3. 4.]

    T = contract_tensors(A, C, 2,1)
    @test T[1,:,:] ≈ [0.4 0.6; 0.4 0.6]

    @test perm_moving_mode(5, 3, 1) == [3,1,2,4,5]
    @test perm_moving_mode(6, 2, 5) == [1,3,4,5,2,6]

    a = 1.0*reshape(collect(1:16), (2,2,2,2))
    b = join_modes(a, 2,4)
    @test vec(a[1,:,1,:]) == b[1,:,1]

    a = ones(5,4,3,2,1)
    b = join_modes(a, 2,3)
    @test size(b) == (5,12,2,1)
    b = join_modes(a, 2,4)
    @test size(b) == (5,8,3,1)
    b = join_modes(a, 1,4)
    @test size(b) == (10,4,3,1)
    b = join_modes(a, 1,5)
    @test size(b) == (5,4,3,2)


    a = 1.0*reshape(collect(1:64), (4,2,2,2,2))
    b = join_modes(a, 2,4)
    @test vec(a[1,:,2,:,1]) == b[1,:,2,1]
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
    add_qubo2graph!(mg_t, qubo)

    @test collect(vertices(mg_t)) == (1:9)

    @test props(mg_t, 1)[:h] == 0.5
    @test props(mg_t, 2)[:h] == 0.5
    @test props(mg_t, 3)[:h] == 0.5
    @test props(mg_t, 4)[:h] == 0.5
    @test props(mg_t, Edge(1,6))[:J] == 0.5

    @test props(mg_t, Edge(1,2))[:side] == ["r", "l"]
    @test props(mg_t, Edge(1,6))[:side] == ["d", "u"]
    @test props(mg_t, Edge(6,1))[:side] == ["d", "u"]
end

@testset "testing tensor generator" begin
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

    @test spins2index(-1) == 1
    @test spins2index(1) == 2
    @test_throws ErrorException spins2index(2)
end

@testset "testing graphical representation" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,4) 0.5]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.5; (4,9) 0.5; (5,5) 0.2; (5,6) 0.5; (5,8) 0.5; (6,6) 0.2; (6,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    mg = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph!(mg, qubo);

    @test bond_directions(mg, 1)  == ["r", "d"]
    @test bond_directions(mg, 5) == ["l", "r", "u", "d"]
    @test bond_directions(mg, 9) == ["l", "u"]


    @test read_pair_from_edge(mg, 1,2, :side) == ["r", "l"]
    @test read_pair_from_edge(mg, 4,5, :side) == ["l", "r"]
    @test read_pair_from_edge(mg, 5,4, :side) == ["r", "l"]
    @test read_pair_from_edge(mg, 3,4, :side) == ["d", "u"]
    @test read_pair_from_edge(mg, 4,3, :side) == ["u", "d"]
    @test_throws ErrorException read_pair_from_edge(mg, 1,3, :side)

    write_pair2edge!(mg, 1,2, :test, ["t1", "t2"])
    write_pair2edge!(mg, 3,2, :test, ["t3", "t2"])

    @test read_pair_from_edge(mg, 1,2, :test) == ["t1", "t2"]
    @test read_pair_from_edge(mg, 2,3, :test) == ["t2", "t3"]

    @test_throws ErrorException write_pair2edge!(mg, 1,3, :test, ["t1", "t3"])
    @test_throws ErrorException write_pair2edge!(mg, 1,2, :test, ["t1", "t2", "t3"])

    @test get_modes(mg, 1) == [2, 4]
    @test get_modes(mg, 5) == [1, 2, 3, 4]
    @test get_modes(mg, 9) == [1, 3]

    @test readJs(mg, 1) == (0.5, 0.5, 0.2)
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
    mg2 = make_graph3x3();
    qubo = make_qubo();
    add_qubo2graph!(mg, qubo);
    add_qubo2graph!(mg1, qubo);
    add_qubo2graph!(mg2, qubo);

    @test props(mg, Edge(4,5))[:modes] == [0,0]

    @test props(mg, Edge(4,5))[:side] == ["l", "r"]
    @test props(mg, Edge(5,2))[:side] == ["d", "u"]
    @test props(mg, Edge(1,2))[:side] == ["r", "l"]

    @test_throws ErrorException add_tensor2vertex(mg, 1, 3)

    set_spins2firs_k!(mg)
    s = [1,1,1,-1,-1,-1,1,1,1]
    set_spins2firs_k!(mg1, s)
    set_spins2firs_k!(mg2, 1)

    @test props(mg, Edge(4,5))[:modes] == [1,2]
    @test props(mg, Edge(1,2))[:modes] == [1,1]
    @test props(mg, Edge(2,5))[:modes] == [3,3]
    @test props(mg, Edge(2,3))[:modes] == [2,1]
    @test props(mg, Edge(3,4))[:modes] == [2,2]

    @test props(mg1, Edge(4,5))[:modes] == [1,2]
    @test props(mg2, Edge(1,2))[:modes] == [1,1]

    @test norm(props(mg, 5)[:tensor] - props(mg1, 5)[:tensor]) > 2.

    @test props(mg1, 1)[:tensor] ≈ props(mg2, 1)[:tensor]
    @test props(mg, 2)[:tensor] ≈ props(mg2, 2)[:tensor]
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
    add_qubo2graph!(mg, qubo);

    set_spins2firs_k!(mg)

    contract_vertices!(mg, 5,8)
    contract_vertices!(mg, 6,7)
    contract_vertices!(mg, 4,9)
    T = props(mg, 5)[:tensor]
    @test T[1,1,1,1,1] == 0.33287108369807955
    @test T[1,2,1,2,1] == 4.481689070338066
    @test ndims(T) == 5
    @test ndims(props(mg, 4)[:tensor]) == 3
    @test props(mg, Edge(5,6))[:modes] == [1,1,4,3]
    @test props(mg, Edge(5,4))[:modes] == [1,2,3,5]
    @test props(mg, Edge(5,2))[:modes] == [3,3]

    @test length(collect(edges(mg))) == 7
end


@testset "test partition function" begin

    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.; (1,6) 0.; (2,2) 0.2; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.; (4,9) 0.; (5,5) 0.2; (5,6) 0.; (5,8) 0.; (6,6) 0.2; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    mg = make_graph3x3();
    qubo = make_qubo()
    add_qubo2graph!(mg, qubo);

    set_spins2firs_k!(mg)

    contract_vertices!(mg, 5,8)
    contract_vertices!(mg, 6,7)
    contract_vertices!(mg, 4,9)
    T = props(mg, 5)[:tensor]

    @test props(mg, Edge(5,6))[:modes] == [1,1,4,3]
    @test props(mg, Edge(5,4))[:modes] == [1,2,3,5]
    @test props(mg, Edge(5,2))[:modes] == [3,3]

    combine_legs_exact!(mg, 5,6)

    T2 = props(mg, 5)[:tensor]
    @test T[1,1,1,1,:] ≈ T2[1,1,1,:]

    combine_legs_exact!(mg, 4,5)

    T1 = props(mg, 5)[:tensor]
    @test T2[1,1,:,1] ≈ T1[1,1,:]

    contract_vertices!(mg, 1,6)
    contract_vertices!(mg, 2,5)
    contract_vertices!(mg, 3,4)

    combine_legs_exact!(mg, 1,2)
    combine_legs_exact!(mg, 2,3)

    contract_vertices!(mg, 2,3)

    contract_vertices!(mg, 1,2)

    @test props(mg, 1)[:tensor][1] ≈ (exp(0.2)+exp(-0.2))^9

    # testing move_modes!

    mg1 = make_graph3x3();
    qubo = make_qubo()
    add_qubo2graph!(mg1, qubo);
    set_spins2firs_k!(mg1)

    contract_vertices!(mg1, 5,8)
    contract_vertices!(mg1, 6,7)
    contract_vertices!(mg1, 4,9)

    modes = props(mg1, Edge(5,6))[:modes]
    @test modes == [1,1,4,3]
    move_modes!(mg1, 5, 3)
    @test modes == [1,1,3,3]
    move_modes!(mg1, 6, 2)
    @test modes == [1,1,3,2]

    @testset "further testing" begin

        function proceed(qubo::Vector{Qubo_el})

            mg = make_graph3x3();
            add_qubo2graph!(mg, qubo);

            set_spins2firs_k!(mg)

            contract_vertices!(mg, 5,8)
            contract_vertices!(mg, 6,7)
            contract_vertices!(mg, 4,9)

            combine_legs_exact!(mg, 5,6)
            combine_legs_exact!(mg, 4,5)

            contract_vertices!(mg, 1,6)
            contract_vertices!(mg, 2,5)
            contract_vertices!(mg, 3,4)

            combine_legs_exact!(mg, 1,2)
            combine_legs_exact!(mg, 2,3)

            contract_vertices!(mg, 2,3)
            contract_vertices!(mg, 1,2)

            return props(mg, 1)[:tensor][1]
        end

        function make_qubo()
            qubo = [(1,1) 0.9; (1,2) 0.; (1,6) 0.; (2,2) 0.2; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
            qubo = vcat(qubo, [(4,4) 0.5; (4,5) 0.; (4,9) 0.; (5,5) 0.2; (5,6) 0.; (5,8) 0.; (6,6) 0.2; (6,7) 0.])
            qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) 0.2])
            [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
        end

        @test proceed(make_qubo()) ≈ (exp(0.2)+exp(-0.2))^7*(exp(0.5)+exp(-0.5))*(exp(0.9)+exp(-0.9))

        function make_qubo1()
            qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.; (2,2) 0.2; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
            qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.; (4,9) 0.; (5,5) 0.2; (5,6) 0.5; (5,8) 0.; (6,6) 0.2; (6,7) 0.])
            qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) 0.6])
            [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
        end

        val = (exp(0.2)+exp(-0.2))^4*(exp(0.6)+exp(-0.6))*(exp(-0.5)*exp(0.2)^2+2*exp(0.5)+exp(-0.5)*exp(-0.2)^2)^2
        @test proceed(make_qubo1()) ≈ val
    end
end

@testset "particular configuration" begin
    function proceed(qubo::Vector{Qubo_el}, s::Int)

        mg = make_graph3x3();
        add_qubo2graph!(mg, qubo);

        set_spins2firs_k!(mg, fill(s,9))

        contract_vertices!(mg, 5,8)
        contract_vertices!(mg, 6,7)
        contract_vertices!(mg, 4,9)

        combine_legs_exact!(mg, 5,6)
        combine_legs_exact!(mg, 4,5)

        contract_vertices!(mg, 1,6)
        contract_vertices!(mg, 2,5)
        contract_vertices!(mg, 3,4)

        combine_legs_exact!(mg, 1,2)
        combine_legs_exact!(mg, 2,3)

        contract_vertices!(mg, 2,3)
        contract_vertices!(mg, 1,2)

        return props(mg, 1)[:tensor][1]
    end

    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.; (1,6) 0.; (2,2) 0.2; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) 0.; (4,9) 0.; (5,5) 0.2; (5,6) 0.; (5,8) 0.; (6,6) 0.2; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    @test proceed(make_qubo(), 1) ≈ exp(-0.2)^9

    function make_qubo1()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 0.; (2,2) 0.3; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) -0.7; (4,9) 0.; (5,5) 0.2; (5,6) -0.9; (5,8) 0.; (6,6) 0.2; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) -0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    @test proceed(make_qubo1(), 1) ≈ exp(-0.2)^6*exp(-0.3)*exp(-0.5)*exp(0.7)*exp(0.9)
    @test proceed(make_qubo1(), -1) ≈ exp(0.2)^6*exp(0.3)*exp(-0.5)*exp(0.7)*exp(0.9)

end

@testset "svd approximation of connections" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.7; (1,6) 0.3; (2,2) -0.2; (2,3) 0.1; (2,5) 0.; (3,3) 0.2; (3,4) 0.]
        qubo = vcat(qubo, [(4,4) 0.2; (4,5) -0.8; (4,9) 1.9; (5,5) 1.2; (5,6) -0.5; (5,8) 0.99; (6,6) 0.2; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) -1.2; (8,8) 0.2; (8,9) 1.0; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    mg = make_graph3x3();
    qubo = make_qubo()
    add_qubo2graph!(mg, qubo);

    mg_exact = make_graph3x3();
    qubo = make_qubo()
    add_qubo2graph!(mg_exact, qubo);

    set_spins2firs_k!(mg)
    set_spins2firs_k!(mg_exact)

    contract_vertices!(mg, 5,8)
    contract_vertices!(mg, 6,7)
    contract_vertices!(mg, 4,9)

    combine_legs_exact!(mg, 5,6)
    combine_legs_exact!(mg, 4,5)

    contract_vertices!(mg_exact, 5,8)
    contract_vertices!(mg_exact, 6,7)
    contract_vertices!(mg_exact, 4,9)

    combine_legs_exact!(mg_exact, 5,6)
    combine_legs_exact!(mg_exact, 4,5)

    reduce_bond_size_svd!(mg, 4,5)
    reduce_bond_size_svd!(mg, 5,6)
    reduce_bond_size_svd!(mg, 6,5)
    reduce_bond_size_svd!(mg, 5,4)

    t6 = props(mg, 6)[:tensor]
    t5 = props(mg, 5)[:tensor]
    t4 = props(mg, 4)[:tensor]

    T6 = props(mg_exact, 6)[:tensor]
    T5 = props(mg_exact, 5)[:tensor]
    T4 = props(mg_exact, 4)[:tensor]

    T = contract_tensors(T4, T5, 1,2)
    a = contract_tensors(T, T6, 2,1)

    t = contract_tensors(t4, t5, 1,2)
    b = contract_tensors(t, t6, 2,1)

    @test norm(abs.(a-b)) < 1e-11

    contract_vertices!(mg, 1,6)
    contract_vertices!(mg, 2,5)
    contract_vertices!(mg, 3,4)

    combine_legs_exact!(mg, 1,2)
    combine_legs_exact!(mg, 2,3)

    contract_vertices!(mg_exact, 1,6)
    contract_vertices!(mg_exact, 2,5)
    contract_vertices!(mg_exact, 3,4)

    combine_legs_exact!(mg_exact, 1,2)
    combine_legs_exact!(mg_exact, 2,3)

    reduce_bond_size_svd!(mg, 3,2)
    reduce_bond_size_svd!(mg, 2,1)
    reduce_bond_size_svd!(mg, 1,2)
    reduce_bond_size_svd!(mg, 2,3)

    contract_vertices!(mg, 2,3)
    contract_vertices!(mg, 1,2)

    contract_vertices!(mg_exact, 2,3)
    contract_vertices!(mg_exact, 1,2)

    @test props(mg, 1)[:tensor][1] - props(mg_exact, 1)[:tensor][1] < 1e-10

    mg = make_graph3x3();
    add_qubo2graph!(mg, qubo);
    @test compute_marginal_prob(mg, Int[]) ≈ props(mg, 1)[:tensor][1]

    mg = make_graph3x3();
    add_qubo2graph!(mg, qubo);
    @test compute_marginal_prob(mg, Int[], false) ≈ props(mg_exact, 1)[:tensor][1]
end

@testset "testing axiliary functions of the solver" begin
    conf = [1 1; 1 -1; -1 1; -1 -1]
    @test add_another_spin2configs(conf) == [1 1 -1; 1 -1 -1; -1 1 -1; -1 -1 -1; 1 1 1; 1 -1 1; -1 1 1; -1 -1 1]
    @test last_m_els([1,2,3,4], 2) == [3,4]
    @test last_m_els([1,2,3,4], 5) == [1,2,3,4]
end

@testset "solving simplest train problem" begin
    # simplest train problem, small example in the train paper
    #two trains approaching the single segment in opposite directions
    # 4 logireduce_bond_size_svd!

    #grig embedding_scheme (a embede all q-bits to remove artificial degeneration)
    # logical -> physical
    #1 -> 2
    # 2 -> 1,6
    # 3 -> 3,4
    # 4 -> 5

    #Jii
    # logical -> physical
    # 1 -> 2 -1.75
    # 2 -> 1 -1.25
    # 3 -> 3 -1.75
    # 4 -> 5 -0.75
    # (none) -> 7,8,9 0.1 to remove degeneration

    #Jij
    # logical -> physical
    #(1,2) -> (1,2) 1.75
    #(1,3) -> (2,3) 1.75
    # (2,4) -> (5,6) 1.75
    #(3,4)  -> (4,5) 1.75

    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,6) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,4) css]
        qubo = vcat(qubo, [(4,4) 0.; (4,5) 1.75; (4,9) 0.; (5,5) -0.75; (5,6) 1.75; (5,8) 0.; (6,6) 0.; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()

    @test optimisation_step_naive(train_qubo, [1]) ≈ 4.417625562495993e7
    @test optimisation_step_naive(train_qubo, [-1]) ≈ 1.6442466626666823e7
    conf, f = naive_solve(train_qubo, 2, true)

    #logical 1st [1,0,0,1]
    # 2 -> +1
    # 1,6 -> -1
    # 3,4 -> -1
    # 5, -> 1
    # 7,8,9 artifially set to 1

    # physical first [-1,1,-1,-1,1,-1,1,1,1]

    #logical ground [0,1,1,0]

    # 2 -> -1
    # 1,6 -> +1
    # 3,4 -> +1
    # 5 -> -1
    # 7,8,9 artifially set to 1

    # physical ground [1,-1,1,1,-1,1,1,1,1]

    @test conf[1,:] == [-1,1,-1,-1,1,-1,1,1,1]
    @test conf[2,:] == [1,-1,1,1,-1,1,1,1,1]

    # end exact calculation without svd approximation

    conf, f = naive_solve(train_qubo, 2, false)

    @test conf[1,:] == [-1,1,-1,-1,1,-1,1,1,1]
    @test conf[2,:] == [1,-1,1,1,-1,1,1,1,1]

    # here we give a little Jii to 7,8,9 q-bits to allow there for 8 additional
    # combinations with low excitiation energies

    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,6) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,4) css]
        qubo = vcat(qubo, [(4,4) 0.; (4,5) 1.75; (4,9) 0.; (5,5) -0.75; (5,6) 1.75; (5,8) 0.; (6,6) 0.; (6,7) 0.])
        qubo = vcat(qubo, [(7,7) -0.1; (7,8) 0.; (8,8) -0.1; (8,9) 0.; (9,9) -0.1])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    permuted_train_qubo = make_qubo()

    conf, f = naive_solve(permuted_train_qubo, 16, true)
    # this correspond to the ground
    for i in 9:16
        @test conf[i, 1:6] == [1,-1,1,1,-1,1]
    end
    # and this to 1st excited
    for i in 1:8
        @test conf[i, 1:6] == [-1,1,-1,-1,1,-1]
    end
end

end

@testset "construction of PEPS" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,6) 1.5; (2,2) 0.6; (2,3) 1.5; (2,5) 0.5; (3,3) 0.2; (3,4) -1.5]
        qubo = vcat(qubo, [(4,4) 2.2; (4,5) 0.25; (4,9) 0.52; (5,5) -0.2; (5,6) -0.5; (5,8) -0.5; (6,6) 2.2; (6,7) 0.01])
        qubo = vcat(qubo, [(7,7) -0.2; (7,8) -0.5; (8,8) 0.2; (8,9) 0.05; (9,9) 0.8])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    function contract_on_graph!(mg)
        contract_vertices!(mg, 5,8)
        contract_vertices!(mg, 6,7)
        contract_vertices!(mg, 4,9)

        combine_legs_exact!(mg, 5,6)
        combine_legs_exact!(mg, 4,5)

        contract_vertices!(mg, 1,6)
        contract_vertices!(mg, 2,5)
        contract_vertices!(mg, 3,4)

        combine_legs_exact!(mg, 1,2)
        combine_legs_exact!(mg, 2,3)

        contract_vertices!(mg, 2,3)
        contract_vertices!(mg, 1,2)
    end

    qubo = make_qubo()

    struct_M = [1 2 3; 6 5 4; 7 8 9]

    @test JfromQubo_el(qubo, 1,2) == 0.5
    @test JfromQubo_el(qubo, 2,1) == 0.5
    @test_throws BoundsError JfromQubo_el(qubo, 1,3)

    @test make_tensor_sizes(false, false, true, true , 2,2) == (1,1,2,2,2)
    @test make_tensor_sizes(true, false, true, true , 2,2) == (2,1,2,2,2)
    @test make_tensor_sizes(false, false, true, false , 2,2) == (1,1,2,1,2)

    mg = make_graph3x3()
    add_qubo2graph!(mg, qubo)

    T = make_peps_node(struct_M, qubo, 1)
    Tp = makeTensor(mg, 1)
    T1 = reshape(Tp, (1,2,1,2,2))
    @test T1 == T

    T = make_peps_node(struct_M, qubo, 5)
    T5 = makeTensor(mg, 5)
    @test T5 == T

    T = make_peps_node(struct_M, qubo, 8)
    Tp = makeTensor(mg, 8)
    T8 = reshape(Tp, (2,2,2,1,2))
    @test T8 == T

    M = make_pepsTN(struct_M, qubo)
    @test M[3,2] == T8
    @test M[2,2] == T5
    @test M[1,1] == T1

    @testset "contracting peps" begin

        a = ones(2,1,1,3)
        b = ones(2,1,3,1)
        e = ones(1,1)
        @test scalar_prod_step(b,a,e) == [3. 3. ; 3. 3.]

        mps = trace_all_spins(M[3,:])
        mpo = trace_all_spins(M[2,:])
        mps1 = trace_all_spins(M[1,:])

        #graphical trace
        set_spins2firs_k!(mg)

        @test mps1[3] == reshape(props(mg, 3)[:tensor], (2,1,1,2))
        @test mpo[1] == reshape(props(mg, 6)[:tensor], (1,2,2,2))
        @test mpo[2] == props(mg, 5)[:tensor]
        @test mps[2] == reshape(props(mg, 8)[:tensor], (2,2,2,1))
        contract_on_graph!(mg)

        #this need to be tested
        mps_t = make_lower_mps(M, 1)
        println(mps_t)

        mps_r = make_lower_mps(M, 2)
        sp = compute_scalar_prod(mps_r, mps1)

        spp = comp_marg_p_first(mps_r, M[1,:], [0,0,0])
        @test sp == props(mg, 1)[:tensor][1]
        @test spp == props(mg, 1)[:tensor][1]

        mg = make_graph3x3()
        add_qubo2graph!(mg, qubo)
        set_spins2firs_k!(mg, [-1,1])
        contract_on_graph!(mg)

        mps11, _ = set_spins_on_mps(M[1,:], [-1, 1, 0])
        sp = compute_scalar_prod(mps_r, mps11)
        spp = comp_marg_p_first(mps_r, M[1,:], [-1,1,0])
        @test sp == props(mg, 1)[:tensor][1]
        @test spp == props(mg, 1)[:tensor][1]

        mg = make_graph3x3()
        add_qubo2graph!(mg, qubo)
        set_spins2firs_k!(mg, [-1,1,1])
        contract_on_graph!(mg)

        mps12, _ = set_spins_on_mps(M[1,:], [-1, 1, 1])
        sp1 = compute_scalar_prod(mps_r, mps12)
        @test sp1 == props(mg, 1)[:tensor][1]

        mg = make_graph3x3()
        add_qubo2graph!(mg, qubo)
        set_spins2firs_k!(mg, [-1,1,1,-1, 1])
        contract_on_graph!(mg)

        mpo3, ses = set_spins_on_mps(M[2,:], [0,1,-1])

        mps_r2 = MPSxMPO(mps, mpo3)

        reduce_bonds_horizontally!(mps12, ses)

        sp3 = compute_scalar_prod(mps_r2, mps12)
        sp4 = comp_marg_p(mps12, mps, M[2,:], [0,1,-1])
        @test sp3 ≈ props(mg, 1)[:tensor][1]
        @test sp3 ≈ sp4

        mg = make_graph3x3()
        add_qubo2graph!(mg, qubo)
        set_spins2firs_k!(mg, [-1,1,1,-1, 1,1,-1])
        contract_on_graph!(mg)

        mps, _ = set_spins_on_mps(M[1,:], [-1,1,1])
        mpo, s = set_spins_on_mps(M[2,:], [1,1,-1])
        reduce_bonds_horizontally!(mps, s)
        mps1 = MPSxMPO(mpo, mps)
        pss = comp_marg_p_last(mps1, M[3,:], [-1,0,0])
        psss = comp_marg_p_last(mps1, M[3,:], [-1,1,0])
        pssss = comp_marg_p_last(mps1, M[3,:], [-1,1,1])
        @test pss ≈ props(mg, 1)[:tensor][1]

        mg = make_graph3x3()
        add_qubo2graph!(mg, qubo)
        set_spins2firs_k!(mg, [-1,1,1,-1, 1,1,-1,1,1])
        contract_on_graph!(mg)

        @test pssss ≈ props(mg, 1)[:tensor][1]
    end
end
