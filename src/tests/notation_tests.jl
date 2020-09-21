
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
