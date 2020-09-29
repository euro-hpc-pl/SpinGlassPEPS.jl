function make_qubo0()
    qubo = [(1,1) -0.2; (1,2) -0.5; (1,4) -1.5; (2,2) -0.6; (2,3) -1.5; (2,5) -0.5; (3,3) -0.2; (3,6) 1.5]
    qubo = vcat(qubo, [(6,6) -2.2; (5,6) -0.25; (6,9) -0.52; (5,5) 0.2; (4,5) 0.5; (5,8) 0.5; (4,4) -2.2; (4,7) -0.01])
    qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) -0.2; (8,9) -0.05; (9,9) -0.8])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "test notation" begin

    @testset "Qubo_el type" begin
        el = Qubo_el((1,2), 1.1)
        @test el.ind == (1,2)
        @test el.coupling == 1.1

        el = Qubo_el{BigFloat}((1,2), 1.1)
        @test el.coupling == 1.1
        @test typeof(el.coupling) == BigFloat

        qubo = make_qubo0()

        @test JfromQubo_el(qubo, 1,2) == -0.5
        @test JfromQubo_el(qubo, 2,1) == -0.5
        @test_throws BoundsError JfromQubo_el(qubo, 1,3)
    end

    @testset "operations on tensors" begin
        A = ones(2,2,2,2)
        @test sum_over_last(A) == 2*ones(2,2,2)
        #@test set_last(A, -1) == ones(2,2,2)

        @test delta(0,-1) == 1
        @test delta(-1,1) == 0
        @test delta(1,1) == 1
        @test c(0, 1., 20, 1.) == 1
        @test Tgen(0,0,0,0,-1,0.,0.,1., 1.) == exp(-1.)
        @test Tgen(0,0,0,0,-1,0.,0.,1., 2.) == exp(-2.)
        @test Tgen(0,0,0,0,-1,0.,0.,-1., 2.) == exp(2.)
    end

    @testset "axiliary" begin
        @test last_m_els([1,2,3,4], 2) == [3,4]
        @test last_m_els([1,2,3,4], 5) == [1,2,3,4]
        @test ind2spin(1) == -1
        @test ind2spin(2) == 1
    end
end
