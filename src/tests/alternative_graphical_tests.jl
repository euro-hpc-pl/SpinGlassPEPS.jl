

@testset "graphical implementation" begin
    function make_qubo()
        qubo = [(1,1) 0.2; (1,2) 0.5; (1,4) 0.5; (2,2) 0.2; (2,3) 0.5; (2,5) 0.5; (3,3) 0.2; (3,6) 0.5]
        qubo = vcat(qubo, [(6,6) 0.2; (6,5) 0.5; (6,9) 0.5; (5,5) 0.2; (4,5) 0.5; (5,8) 0.5; (4,4) 0.2; (4,7) 0.5])
        qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) 0.2; (8,9) 0.5; (9,9) 0.2])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end

    @testset "axiliary functions" begin

        @test sort2lrud(["d", "l"]) == ["l", "d"]
        @test sort2lrud(["u", "d", "l", "r"]) == ["l", "r", "u", "d"]
        @test index2physical(2) == 1
        @test index2physical(1) == -1

        conf = [1 1; 1 -1; -1 1; -1 -1]
        @test add_another_spin2configs(conf) == [1 1 -1; 1 -1 -1; -1 1 -1; -1 -1 -1; 1 1 1; 1 -1 1; -1 1 1; -1 -1 1]
    end

    @testset "operations on modes " begin

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

        a = 1.0*reshape(collect(1:64), (4,2,2,2,2))
        b = join_modes(a, 2,4)
        @test vec(a[1,:,2,:,1]) == b[1,:,2,1]
    end

    @testset "adding qubo to graph" begin
        mg = make_graph3x3()

        qubo = make_qubo()
        add_qubo2graph!(mg, qubo)

        @test collect(vertices(mg)) == (1:9)
        @test props(mg, 1)[:h] == 0.2
        @test props(mg, Edge(1,4))[:J] == 0.5
        @test props(mg, Edge(1,2))[:side] == ["r", "l"]

        @test bond_directions(mg, 1)  == ["r", "d"]
        @test bond_directions(mg, 5) == ["l", "r", "u", "d"]

        @test read_pair_from_edge(mg, 5,4, :side) == ["l", "r"]

        write_pair2edge!(mg, 3,2, :test, ["t3", "t2"])
        @test read_pair_from_edge(mg, 2,3, :test) == ["t2", "t3"]

        @test get_modes(mg, 1) == [2, 4]
        @test get_modes(mg, 5) == [1, 2, 3, 4]
        @test readJs(mg, 2) == (0.5, 0, 0.2)

        T = makeTensor(mg, 9, 1.)
        @test T[1] == 0.4493289641172216

        T3 = [Tgen(l,r,u,d,1,0.5, 0.5, 0.2, 1.)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
        T4 = [Tgen(l,r,u,d,-1,0.5, 0.5, 0.2, 1.)  for d in [-1, 1] for u in [-1, 1] for r in [-1, 1] for l in [-1, 1]]
        T5 = T3+T4
        add_tensor2vertex(mg, 5, 1.)
        @test props(mg, 5)[:tensor] == reshape(T5, (2,2,2,2))

        mg1 = make_graph3x3();
        add_qubo2graph!(mg1, qubo);

        set_spins2firs_k!(mg, Int[], 1.)
        s = [1,1,1,-1,-1,-1,1,1,1]
        set_spins2firs_k!(mg1, s, 1.)

        @test norm(props(mg, 5)[:tensor] - props(mg1, 5)[:tensor]) > 2.

        contract_vertices!(mg, 4,7)
        contract_vertices!(mg, 5,8)
        contract_vertices!(mg, 6,9)

        T = props(mg, 5)[:tensor]
        combine_legs_exact!(mg, 4,5)

        T2 = props(mg, 5)[:tensor]
        @test T[1,1,1,1,:] ≈ T2[1,1,1,:]

        combine_legs_exact!(mg, 5,6)

        T1 = props(mg, 5)[:tensor]
        @test T2[1,1,:,1] ≈ T1[1,1,:]

    end
end

@testset "graphical solving simplest train problem" begin

    @testset "graphical probability computing" begin
        function proceed(qubo::Vector{Qubo_el{Float64}}, s::Int)

            mg = make_graph3x3();
            add_qubo2graph!(mg, qubo);

            set_spins2firs_k!(mg, fill(s,9), 1.)

            contract_vertices!(mg, 4,7)
            contract_vertices!(mg, 5,8)
            contract_vertices!(mg, 6,9)

            combine_legs_exact!(mg, 5,6)
            combine_legs_exact!(mg, 4,5)

            contract_vertices!(mg, 1,4)
            contract_vertices!(mg, 2,5)
            contract_vertices!(mg, 3,6)

            combine_legs_exact!(mg, 1,2)
            combine_legs_exact!(mg, 2,3)

            contract_vertices!(mg, 2,3)
            contract_vertices!(mg, 1,2)

            return props(mg, 1)[:tensor][1]
        end

        function make_qubo1()
            qubo = [(1,1) 0.2; (1,2) 0.5; (1,4) 0.; (2,2) 0.3; (2,3) 0.; (2,5) 0.; (3,3) 0.2; (3,6) 0.]
            qubo = vcat(qubo, [(6,6) 0.2; (5,6) -0.7; (6,9) 0.; (5,5) 0.2; (4,5) -0.9; (5,8) 0.; (4,4) 0.2; (4,7) 0.])
            qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.; (8,8) 0.2; (8,9) 0.; (9,9) -0.2])
            [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
        end

        @test proceed(make_qubo1(), 1) ≈ exp(-0.2)^6*exp(-0.3)*exp(-0.5)*exp(0.7)*exp(0.9)
        @test proceed(make_qubo1(), -1) ≈ exp(0.2)^6*exp(0.3)*exp(-0.5)*exp(0.7)*exp(0.9)
    end

    @testset "real problem with svd approximation and without" begin
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
            qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
            qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
            qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
            [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
        end
        train_qubo = make_qubo()

        @test optimisation_step_naive(train_qubo, [1], 1., 0.) ≈ 4.417625562495993e7
        @test optimisation_step_naive(train_qubo, [-1], 1., 0.) ≈ 1.6442466626666823e7
        conf, f = naive_solve(train_qubo, 2, 1., 1e-12)

        #logical 1st [1,0,0,1]
        # 2 -> 1
        # 1,4 -> -1
        # 3,6 -> -1
        # 5, -> 1
        # 7,8,9 artifially set to 1

        # physical first [-1,1,-1,-1,1,-1,1,1,1]

        #logical ground [0,1,1,0]

        # 2 -> -1
        # 1,4 -> 1
        # 3,6 -> 1
        # 5 -> -1
        # 7,8,9 artifially set to 1

        # physical ground [1,-1,1,1,-1,1,1,1,1]

        @test conf[1,:] == [-1,1,-1,-1,1,-1,1,1,1]
        @test conf[2,:] == [1,-1,1,1,-1,1,1,1,1]

        # end exact calculation without svd approximation

        conf, f = naive_solve(train_qubo, 2, 1., 0.)

        @test conf[1,:] == [-1,1,-1,-1,1,-1,1,1,1]
        @test conf[2,:] == [1,-1,1,1,-1,1,1,1,1]
    end
end

@testset "peps vs graph implementation" begin
    function contract_on_graph!(mg)
        contract_vertices!(mg, 4,7)
        contract_vertices!(mg, 5,8)
        contract_vertices!(mg, 6,9)


        combine_legs_exact!(mg, 5,6)
        combine_legs_exact!(mg, 4,5)

        contract_vertices!(mg, 1,4)
        contract_vertices!(mg, 2,5)
        contract_vertices!(mg, 3,6)

        combine_legs_exact!(mg, 1,2)
        combine_legs_exact!(mg, 2,3)

        contract_vertices!(mg, 2,3)
        contract_vertices!(mg, 1,2)
    end

    qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]
    M = make_pepsTN(grid, qubo, 1.)
    # uses graph notation output for testing
    mg = make_graph3x3()
    add_qubo2graph!(mg, qubo)

    @testset "building" begin
        T = make_peps_node(grid, qubo, 1, 1.)
        Tp = makeTensor(mg, 1, 1.)
        T1 = reshape(Tp, (1,2,1,2,2))
        @test T1 == T

        T = make_peps_node(grid, qubo, 5, 1.)
        T5 = makeTensor(mg, 5, 1.)
        @test T5 == T

        T = make_peps_node(grid, qubo, 8, 1.)
        Tp = makeTensor(mg, 8, 1.)
        T8 = reshape(Tp, (2,2,2,1,2))
        @test T8 == T

        @test M[3,2] == T8
        @test M[2,2] == T5
        @test M[1,1] == T1
    end

    @testset "contracting peps" begin

        a = ones(2,1,1,3)
        b = ones(2,1,3,1)
        e = ones(1,1)

        @test scalar_prod_step(b,a,e) == [3. 3. ; 3. 3.]

        mps = trace_all_spins(M[3,:])
        mpo = trace_all_spins(M[2,:])
        mps1 = trace_all_spins(M[1,:])

        #graphical trace
        set_spins2firs_k!(mg, Int[], 1.)

        @test mps1[3] == reshape(props(mg, 3)[:tensor], (2,1,1,2))
        @test mpo[1] == reshape(props(mg, 4)[:tensor], (1,2,2,2))
        @test mpo[2] == props(mg, 5)[:tensor]
        @test mps[2] == reshape(props(mg, 8)[:tensor], (2,2,2,1))
        contract_on_graph!(mg)

        mps_t = make_lower_mps(M, 1, 0, 0.)
        M1 = mps_t[1]
        M2 = mps_t[2]
        M3 = mps_t[3]

        E = ones(1,1)

        @tensor begin
            D[] := M1[a,x,z,v]*M2[x,y,z,v]*M3[y,a,z1,v1]*E[z1,v1]
        end
        @test D[1] ≈ props(mg, 1)[:tensor][1]
    end
end
