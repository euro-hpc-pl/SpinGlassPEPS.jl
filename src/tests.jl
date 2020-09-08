include("notation.jl")

using Test
using LinearAlgebra


@testset "test axiliary functions" begin

    @testset "Qubo_el type" begin
        el = Qubo_el((1,2), 1.1)
        @test el.ind == (1,2)
        @test el.coupling == 1.1

        el = Qubo_el{BigFloat}((1,2), 1.1)
        @test el.coupling == 1.1
        @test typeof(el.coupling) == BigFloat
    end

    @testset "operations on tensors" begin
        A = ones(2,2,2,2)
        @test sum_over_last(A) == 2*ones(2,2,2)
        @test set_last(A, -1) == ones(2,2,2)

        @test delta(0,-1) == 1
        @test delta(-1,1) == 0
        @test delta(1,1) == 1
        @test c(0, 1., 20, 1.) == 1
        @test Tgen(0,0,0,0,-1,0.,0.,1., 1.) == exp(1.)
        @test Tgen(0,0,0,0,-1,0.,0.,1., 2.) == exp(2.)
        @test Tgen(0,0,0,0,-1,0.,0.,-1., 2.) == exp(-2.)
    end

    @testset "axiliary" begin
        @test spins2index(-1) == 1
        @test spins2index(1) == 2
        @test_throws ErrorException spins2index(2)

        @test last_m_els([1,2,3,4], 2) == [3,4]
        @test last_m_els([1,2,3,4], 5) == [1,2,3,4]
    end
end

include("alternative_approach.jl")


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

include("peps.jl")

function make_qubo()
    qubo = [(1,1) 0.2; (1,2) 0.5; (1,4) 1.5; (2,2) 0.6; (2,3) 1.5; (2,5) 0.5; (3,3) 0.2; (3,6) -1.5]
    qubo = vcat(qubo, [(6,6) 2.2; (5,6) 0.25; (6,9) 0.52; (5,5) -0.2; (4,5) -0.5; (5,8) -0.5; (4,4) 2.2; (4,7) 0.01])
    qubo = vcat(qubo, [(7,7) -0.2; (7,8) -0.5; (8,8) 0.2; (8,9) 0.05; (9,9) 0.8])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "PEPS - axiliary functions" begin
    qubo = make_qubo()

    @test JfromQubo_el(qubo, 1,2) == 0.5
    @test JfromQubo_el(qubo, 2,1) == 0.5
    @test_throws BoundsError JfromQubo_el(qubo, 1,3)

    @test make_tensor_sizes(false, false, true, true , 2,2) == (1,1,2,2,2)
    @test make_tensor_sizes(true, false, true, true , 2,2) == (2,1,2,2,2)
    @test make_tensor_sizes(false, false, true, false , 2,2) == (1,1,2,1,2)

    # partial solution
    ps = Partial_sol{Float64}()
    @test ps.spins == []
    @test ps.objective == 1.
    @test ps.upper_mps == [zeros(0,0,0,0)]

    ps1 = Partial_sol{Float64}([1,1], 1., [ones(2,2,2,2), ones(2,2,2,2)])
    @test ps1.spins == [1,1]
    @test ps1.objective == 1.
    @test ps1.upper_mps == [ones(2,2,2,2), ones(2,2,2,2)]

    ps2 = add_spin(ps1, -1)
    @test ps2.spins == [1,1,-1]
    @test ps2.objective == 0.
    @test ps2.upper_mps == [ones(2,2,2,2), ones(2,2,2,2)]
end

@testset "PEPS network creation" begin

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

        mps_t = make_lower_mps(M, 1, 0.)
        A = mps_t[1]
        B = mps_t[2]
        C = mps_t[3]

        E = ones(1,1)

        @tensor begin
            D[] := A[a,x,z,v]*B[x,y,z,v]*C[y,a,z1,v1]*E[z1,v1]
        end
        @test D[1] == props(mg, 1)[:tensor][1]

        @testset "testing svd approximation" begin

            A1 = copy(A)
            B1 = copy(B)
            C1 = copy(C)

            A2 = copy(A)
            B2 = copy(B)
            C2 = copy(C)


            A,B = reduce_bond_size_svd_right2left(A,B, 1e-12)
            B,C = reduce_bond_size_svd_right2left(B,C, 1e-12)

            B1,C1 = reduce_bond_size_svd_left2right(B1,C1, 1e-12)
            A1,B1 = reduce_bond_size_svd_left2right(A1,B1, 1e-12)


            @tensor begin
                D[] := A[a,x,z,v]*B[x,y,z,v]*C[y,a,z1,v1]*E[z1,v1]
            end

            @test D[1] ≈ props(mg, 1)[:tensor][1]

            @tensor begin
                D1[] := A1[a,x,z,v]*B1[x,y,z,v]*C1[y,a,z1,v1]*E[z1,v1]
            end

            @test D1[1] ≈ props(mg, 1)[:tensor][1]

            mps_svd = svd_approx([A2, B2, C2], 1e-12)
            A2 = mps_svd[1]
            B2 = mps_svd[2]
            C2 = mps_svd[3]

            @test size(A2) == (1,1,1,1)
            @test size(B2) == (1,1,1,1)
            @test size(C2) == (1,1,1,1)

            @tensor begin
                D2[] := A2[a,x,z,v]*B2[x,y,z,v]*C2[y,a,z1,v1]*E[z1,v1]
            end

            @test D2[1] ≈ props(mg, 1)[:tensor][1]

        end

        @testset "testing marginal probabilities for various configurations" begin

            mps_r = make_lower_mps(M, 2, 0.)
            sp = compute_scalar_prod(mps_r, mps1)

            spp, _ = comp_marg_p_first(mps_r, M[1,:], [0,0,0])
            @test sp == props(mg, 1)[:tensor][1]
            @test spp == props(mg, 1)[:tensor][1]

            mg = make_graph3x3()
            add_qubo2graph!(mg, qubo)
            set_spins2firs_k!(mg, [-1,1], 1.)
            contract_on_graph!(mg)

            mps11 = set_spins_on_mps(M[1,:], [-1, 1, 0])
            sp = compute_scalar_prod(mps_r, mps11)

            ##### marginal probability implementation ####
            @test 32. == compute_scalar_prod([ones(1,2,2,1), 2*ones(2,1,2,1)], [ones(1,2,1,2), ones(2,1,1,2)])


            spp, _ = comp_marg_p_first(mps_r, M[1,:], [-1,1,0])
            @test sp ≈ props(mg, 1)[:tensor][1]
            @test spp ≈ props(mg, 1)[:tensor][1]

            mg = make_graph3x3()
            add_qubo2graph!(mg, qubo)
            set_spins2firs_k!(mg, [-1,1,1], 1.)
            contract_on_graph!(mg)

            mps12 = set_spins_on_mps(M[1,:], [-1, 1, 1])
            sp1 = compute_scalar_prod(mps_r, mps12)
            @test sp1 == props(mg, 1)[:tensor][1]

            mg = make_graph3x3()
            add_qubo2graph!(mg, qubo)
            set_spins2firs_k!(mg, [-1,1,1,-1, 1], 1.)
            contract_on_graph!(mg)

            mpo3 = set_spins_on_mps(M[2,:], [-1,1,0])
            mps_r2 = MPSxMPO(mps, mpo3)

            sp3 = compute_scalar_prod(mps_r2, mps12)
            sp4, _ = comp_marg_p(mps12, mps, M[2,:], [-1,1,0])
            @test sp3 ≈ props(mg, 1)[:tensor][1]
            @test sp3 ≈ sp4

            mg = make_graph3x3()
            add_qubo2graph!(mg, qubo)
            set_spins2firs_k!(mg, [-1,1,1,-1, 1,1,-1], 1.)
            contract_on_graph!(mg)

            mps = set_spins_on_mps(M[1,:], [-1,1,1])
            mpo = set_spins_on_mps(M[2,:], [-1,1,1])

            mps1 = MPSxMPO(mpo, mps)
            pss = comp_marg_p_last(mps1, M[3,:], [-1,0,0])
            psss = comp_marg_p_last(mps1, M[3,:], [-1,1,0])
            pssss = comp_marg_p_last(mps1, M[3,:], [-1,1,1])
            @test pss ≈ props(mg, 1)[:tensor][1]

            mg = make_graph3x3()
            add_qubo2graph!(mg, qubo)
            set_spins2firs_k!(mg, [-1,1,1,-1, 1,1,-1,1,1], 1.)
            contract_on_graph!(mg)

            @test pssss ≈ props(mg, 1)[:tensor][1]


            ####   conditional probability implementation
            # TODO more tests

            mps = MPSxMPO([ones(1,2,2,1), 2*ones(2,1,2,1)], [ones(1,2,1,2), ones(2,1,1,2)])
            @test mps == [2*ones(1,4,1,1), 4*ones(4,1,1,1)]

            mps = MPSxMPO([ones(1,2,2,1,2), 2*ones(2,1,2,1,2)], [ones(1,2,1,2), ones(2,1,1,2)])
            @test mps == [2*ones(1,4,1,1,2), 4*ones(4,1,1,1,2)]


            b = compute_scalar_prod([ones(1,2,2,1), ones(2,1,2,1)], [ones(1,2,1,2,2), 2*ones(2,1,1,2)])
            @test b == [32.0, 32.0]

            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1), ones(2,2))
            @test a == [8.0 8.0; 8.0 8.0]
            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1,2), ones(2,2))
            @test a[:,:,1] == [8.0 8.0; 8.0 8.0]
            @test a[:,:,2] == [8.0 8.0; 8.0 8.0]
            a = scalar_prod_step(ones(2,2,1,2), ones(2,2,2,1), ones(2,2,2))
            @test a[:,:,1] == [8.0 8.0; 8.0 8.0]
            @test a[:,:,2] == [8.0 8.0; 8.0 8.0]

            v1 = [ones(1,2,1,2,2), ones(2,2,1,2,2), ones(2,2,1,2,2), ones(2,1,1,2,2)]
            v2 = [ones(1,2,2,1), ones(2,2,2,1), ones(2,2,2,1), ones(2,1,2,1)]
            a = conditional_probabs(v1, v2, [1,1,1])
            @test a == [0.5, 0.5]

            a = chain2point([ones(1,2,1,1,2), ones(2,1,1,1)])
            @test a == [0.5, 0.5]

            a = chain2point([reshape([1.,0.], (1,2,1,1)), ones(2,2,1,1,2), ones(2,2,1,1), ones(2,1,1,1)])
            @test a == [0.5, 0.5]

            a = chain2point([reshape([1.,0.], (1,2,1,1)), ones(2,1,1,1,2)])
            @test a == [0.5, 0.5]
        end
    end
end


@testset "PEPS - solving simple train problem" begin
    # simplest train problem, small example in the train paper
    #two trains approaching the single segment in opposite directions


    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (6,5) 1.75; (6,9) 0.; (5,5) -0.75; (5,4) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()


    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(train_qubo, grid, 4; β = 1.)

    ses1 = solve_arbitrary_decomposition(train_qubo, grid, 4; β = 1.)
    #first
    @test ses[3].spins == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test ses[4].spins == [1,-1,1,1,-1,1,1,1,1]

    @test ses[4].spins == ses1[4].spins
    @test ses[3].spins == ses1[3].spins
    @test ses[2].spins == ses1[2].spins
    @test ses[1].spins == ses1[1].spins

    # here we give a little Jii to 7,8,9 q-bits to allow there for 8 additional
    # combinations with low excitiation energies

    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) 1.75; (6,9) 0.; (5,5) -0.75; (4,5) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) -0.1; (7,8) 0.; (8,8) -0.1; (8,9) 0.; (9,9) -0.1])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    permuted_train_qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(permuted_train_qubo, grid, 16; β = 1.)

    # this correspond to the ground
    for i in 9:16
        @test ses[i].spins[1:6] == [1,-1,1,1,-1,1]
    end

    # and this to 1st excited
    for i in 1:8
        @test ses[i].spins[1:6] == [-1,1,-1,-1,1,-1]
    end

    @testset "svd approximatimation in solution" begin

        ses_a = solve(permuted_train_qubo, grid, 16; β = 1., threshold = 1e-12)

        for i in 9:16
            @test ses_a[i].spins[1:6] == [1,-1,1,1,-1,1]
        end

        # and this to 1st excited
        for i in 1:8
            @test ses_a[i].spins[1:6] == [-1,1,-1,-1,1,-1]
        end
    end
end


@testset "PEPS  - solving it on BigFloat" begin
    T = BigFloat
    function make_qubo()
        css = -2.
        qubo = [(1,1) -1.25; (1,2) 1.75; (1,4) css; (2,2) -1.75; (2,3) 1.75; (2,5) 0.; (3,3) -1.75; (3,6) css]
        qubo = vcat(qubo, [(6,6) 0.; (5,6) 1.75; (6,9) 0.; (5,5) -0.75; (4,5) 1.75; (5,8) 0.; (4,4) 0.; (4,7) 0.])
        qubo = vcat(qubo, [(7,7) css; (7,8) 0.; (8,8) css; (8,9) 0.; (9,9) css])
        [Qubo_el{T}(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()

    grid = [1 2 3; 4 5 6; 7 8 9]

    ses = solve(train_qubo, grid, 4; β = T(2.))

    #first
    @test ses[3].spins == [-1,1,-1,-1,1,-1,1,1,1]
    #ground
    @test ses[4].spins == [1,-1,1,1,-1,1,1,1,1]

    # we should think about this atol

    @test typeof(ses[1].objective) == BigFloat
end

@testset "larger QUBO" begin
    function make_qubo()
        qubo = [(1,1) -1.; (1,2) 0.5; (1,5) 0.5; (2,2) 1.; (2,3) 0.5; (2,6) 0.5; (3,3) -1.0; (3,4) 0.5; (3,7) 0.5; (4,4) 1.0; (4,8) 0.5]
        qubo = vcat(qubo, [(5,5) -1.; (5,6) 0.5; (5,9) 0.5; (6,6) 1.; (6,7) 0.5; (6,10) 0.5; (7,7) -1.0; (7,8) 0.5; (7,11) 0.5; (8,8) 1.0; (8,12) 0.5])
        qubo = vcat(qubo, [(9,9) -1.; (9,10) 0.5; (9,13) 0.5; (10,10) 1.; (10,11) 0.5; (10,14) 0.5; (11,11) -1.0; (11,12) 0.5; (11,15) 0.5; (12,12) 1.0; (12,16) 0.5])
        qubo = vcat(qubo, [(13,13) -1.; (13,14) 0.5; (14,14) 1.; (14,15) 0.5; (15,15) -1.0; (15,16) 0.5; (16,16) 1.0])
        [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
    end
    train_qubo = make_qubo()


    grid = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    @time ses = solve(train_qubo, grid, 10; β = 2., threshold = 1e-12)
    @test ses[end].spins == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
end
