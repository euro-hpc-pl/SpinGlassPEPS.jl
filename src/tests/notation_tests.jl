@testset "node type, connections on the graph" begin

    grid = [1 2 3; 4 5 6; 7 8 9]

    n = Node_of_grid(3, grid)
    @test n.i == 3
    @test n.spin_inds == [3]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[3,2]]
    @test n.right == Array{Int64,1}[]
    @test n.up == Array{Int64,1}[]
    @test n.down == [[3,6]]

    n = Node_of_grid(5, grid)
    @test n.i == 5
    @test n.spin_inds == [5]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[5,4]]
    @test n.right == [[5,6]]
    @test n.up == [[5,2]]
    @test n.down == [[5,8]]

    grid1 = [1 2 3 4; 5 6 7 8; 9 10 11 12]
    n = Node_of_grid(4, grid1)
    @test n.i == 4
    @test n.spin_inds == [4]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[4,3]]
    @test n.right == Array{Int64,1}[]
    @test n.up == Array{Int64,1}[]
    @test n.down == [[4,8]]


    n = Node_of_grid(9, grid1)
    @test n.i == 9
    @test n.spin_inds == [9]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == Array{Int64,1}[]
    @test n.right == [[9,10]]
    @test n.up == [[9,5]]
    @test n.down == Array{Int64,1}[]

    n = Node_of_grid(12, grid1)
    @test n.i == 12
    @test n.spin_inds == [12]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[12,11]]
    @test n.right == Array{Int64,1}[]
    @test n.up == [[12,8]]
    @test n.down == Array{Int64,1}[]
end


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
        @test ind2spin(1,2) == [-1]
        @test ind2spin(2,2) == [1]
        @test ind2spin(1,1)[1] == 0

        @test ind2spin(1, 16) == [-1,-1,-1,-1]
        @test ind2spin(2, 16) == [1,-1,-1,-1]
        @test ind2spin(3, 16) == [-1,1,-1,-1]

        @test spins2ind([-1,-1,-1,-1]) == 1
        @test spins2ind([1,-1,-1,-1]) == 2
        @test spins2ind([-1,1,-1,-1]) == 3

        s = ind2spin(7, 16)
        @test spins2ind(s) == 7

        s = ind2spin(9, 16)
        @test spins2ind(s) == 9

        s = ind2spin(12, 16)
        @test spins2ind(s) == 12

        s = ind2spin(12, 64)
        @test spins2ind(s) == 12

    end
end
