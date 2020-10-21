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
    @test n.connected_nodes == [2,6]
    @test n.connected_spins == [[3 2], [3 6]]

    n = Node_of_grid(5, grid)
    @test n.i == 5
    @test n.spin_inds == [5]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[5,4]]
    @test n.right == [[5,6]]
    @test n.up == [[5,2]]
    @test n.down == [[5,8]]
    @test n.connected_spins == [[5 4], [5 6], [5 2], [5 8]]

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

    ns = [Node_of_grid(i, grid1) for i in 1:maximum(grid1)]
    @test get_system_size(ns) == 12


    grid = Array{Array{Int}}(undef, (2,2))
    grid[1,1] = [1 2;5 6]
    grid[1,2] = [3 4; 7 8]
    grid[2,1] = [9 10;13 14]
    grid[2,2] = [11 12;15 16]
    grid = Array{Array{Int}}(grid)
    M = [1 2;3 4]

    n = Node_of_grid(1,M, grid)
    @test n.i == 1
    @test n.spin_inds == [1, 2, 5, 6]
    @test n.intra_struct == [[1, 2], [5, 6], [1, 5], [2, 6]]
    @test n.left == Array{Int64,1}[]
    @test n.right == [[2, 3], [6, 7]]
    @test n.up == Array{Int64,1}[]
    @test n.down == [[5, 9], [6, 10]]
    @test n.connected_nodes == [2, 3]
    @test n.connected_spins == [[2 3; 6 7], [5 9; 6 10]]


    nc = Node_of_grid(1,M, grid; chimera = true)
    @test nc.i == 1
    @test nc.spin_inds == [1, 2, 5, 6]
    @test nc.intra_struct == [[1, 2], [1, 6], [5, 2], [5, 6]]
    @test nc.left == Array{Int64,1}[]
    @test nc.right == [[2, 4], [6, 8]]
    @test nc.up == Array{Int64,1}[]
    @test nc.down == [[1, 9], [5, 13]]
    @test nc.connected_nodes == [2,3]

    nc = Node_of_grid(2,M, grid; chimera = true)
    @test nc.intra_struct == [[3, 4], [3, 8], [7, 4], [7, 8]]
    @test nc.left == [[4, 2], [8, 6]]
    @test nc.right == Array{Int64,1}[]
    @test nc.up == Array{Int64,1}[]
    @test nc.down == [[3, 11], [7, 15]]


    nc = Node_of_grid(4,M, grid; chimera = true)
    @test nc.intra_struct == [[11, 12], [11, 16], [15, 12], [15, 16]]
    @test nc.left == [[12, 10], [16, 14]]
    @test nc.right == Array{Int64,1}[]
    @test nc.up == [[11, 3], [15, 7]]
    @test nc.down == Array{Int64,1}[]


    # larger chimera

    grid = Array{Array{Int}}(undef, (1,2))
    grid[1,1] = [1 2; 5 6 ; 9 10 ; 13 14]
    grid[1,2] = [3 4; 7 8; 11 12; 15 16]
    grid = Array{Array{Int}}(grid)
    M = reshape([1 2], (1,2))

    nc_l = Node_of_grid(1,M, grid; chimera = true)

    @test nc_l.intra_struct == [[1, 2], [1, 6], [1, 10], [1, 14], [5, 2], [5, 6], [5, 10], [5, 14], [9, 2], [9, 6], [9, 10], [9, 14], [13, 2], [13, 6], [13, 10], [13, 14]]
    @test nc_l.right == [[2, 4], [6, 8], [10, 12], [14, 16]]

    @test nc_l.connected_spins[1][:,1] == [2, 6, 10, 14]
    @test nc_l.connected_spins[1][:,2] == [4, 8, 12, 16]

    # chimera node 2 x 2

    grid = Array{Array{Int}}(undef, (1,1))
    grid[1,1] = [1 2;3 4]
    grid = Array{Array{Int}}(grid)
    M = reshape([1], (1,1))
    n = Node_of_grid(1,M, grid; chimera = true)
    @test n.intra_struct == [[1, 2], [1, 4], [3, 2], [3, 4]]

    # chimera node 2 x 8

    grid1 = Array{Array{Int}}(undef, (1,1))
    grid1[1,1] = [1 2;3 4; 5 6; 7 8]
    grid1 = Array{Array{Int}}(grid1)
    M = reshape([1], (1,1))
    n = Node_of_grid(1,M, grid1; chimera = true)
    @test n.intra_struct == [[1, 2], [1, 4], [1, 6], [1, 8], [3, 2], [3, 4], [3, 6], [3, 8], [5, 2], [5, 4], [5, 6], [5, 8], [7, 2], [7, 4], [7, 6], [7, 8]]


    grid = Array{Array{Int}}(undef, (3,3))
    grid[1,1] = [1 2;6 7]
    grid[1,2] = [3 4; 8 9]
    grid[1,3] = reshape([5 ; 10], (2,1))
    grid[2,1] = [11 12;16 17]
    grid[2,2] = [13 14;18 19]
    grid[2,3] = reshape([15; 20], (2,1))
    grid[3,1] = reshape([21; 22], (1,2))
    grid[3,2] = reshape([23; 24], (1,2))
    grid[3,3] = reshape([25], (1,1))

    grid = Array{Array{Int}}(grid)
    M = [1 2 3 ;4 5 6; 7 8 9]

    n = Node_of_grid(5,M, grid)
    @test n.i == 5
    @test n.spin_inds == [13, 14, 18, 19]
    @test n.intra_struct == [[13, 14], [18, 19], [13, 18], [14, 19]]
    @test n.left == [[13, 12], [18, 17]]
    @test n.right == [[14, 15], [19, 20]]
    @test n.up == [[13, 8], [14, 9]]
    @test n.down == [[18, 23], [19, 24]]
    @test n.connected_nodes == [4, 6, 2, 8]

    n = Node_of_grid(9,M, grid)
    @test n.i == 9
    @test n.spin_inds == [25]
    @test n.intra_struct == Array{Int64,1}[]
    @test n.left == [[25, 24]]
    @test n.right == Array{Int64,1}[]
    @test n.up == [[25, 20]]
    @test n.down == Array{Int64,1}[]
    @test n.connected_nodes == [8,6]

    ns = [Node_of_grid(i, M, grid) for i in 1:maximum(M)]

    @test get_system_size(ns) == 25

end


function make_qubo0()
    qubo = [(1,1) -0.2; (1,2) -0.5; (1,4) -1.5; (2,2) -0.6; (2,3) -1.5; (2,5) -0.5; (3,3) -0.2; (3,6) 1.5]
    qubo = vcat(qubo, [(6,6) -2.2; (5,6) -0.25; (6,9) -0.52; (5,5) 0.2; (4,5) 0.5; (5,8) 0.5; (4,4) -2.2; (4,7) -0.01])
    qubo = vcat(qubo, [(7,7) 0.2; (7,8) 0.5; (8,8) -0.2; (8,9) -0.05; (9,9) -0.8])
    [Qubo_el(qubo[i,1], qubo[i,2]) for i in 1:size(qubo, 1)]
end

@testset "axiliary on qubo" begin
    qubo = make_qubo0()
    n = Node_of_grid(1, qubo)
    @test n.connected_nodes == [2,4]

    n = Node_of_grid(5, qubo)
    @test n.connected_nodes == [2,6,4,8]

    n = Node_of_grid(9, qubo)
    @test n.connected_nodes == [6,8]

    @test get_system_size(qubo) == 9

    M = rand(10,10)
    m = matrix2qubo_vec(M)

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

        @test reindex(1, [1,2,3,4,5], [2,3]) == 1
        @test reindex(2, [1,2,3,4,5], [2,3]) == 1
        @test reindex(3, [1,2,3,4,5], [2,3]) == 2
        @test reindex(4, [1,2,3,4,5], [2,3]) == 2
        @test reindex(7, [1,2,3,4,5], [2,3]) == 4
        @test reindex(8, [1,2,3,4,5], [2,3]) == 4
        @test reindex(10, [1,2,3,4,5], [2,3]) == 1

    end
end
