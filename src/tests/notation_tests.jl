function make_interactions()
    J_h = [(1,1) -0.2; (1,2) -0.5; (1,4) -1.5; (2,2) -0.6; (2,3) -1.5; (2,5) -0.5; (3,3) -0.2; (3,6) 1.5]
    J_h = vcat(J_h, [(6,6) -2.2; (5,6) -0.25; (6,9) -0.52; (5,5) 0.2; (4,5) 0.5; (5,8) 0.5; (4,4) -2.2; (4,7) -0.01])
    J_h = vcat(J_h, [(7,7) 0.2; (7,8) 0.5; (8,8) -0.2; (8,9) -0.05; (9,9) -0.8])
    [Interaction(J_h[i,1], J_h[i,2]) for i in 1:size(J_h, 1)]
end


@testset "axiliary on interactions" begin
    M = ones(2,2)
    interactions = M2interactions(M)
    @test interactions == Interaction{Float64}[Interaction{Float64}((1, 1), 1.0), Interaction{Float64}((1, 2), 1.0), Interaction{Float64}((2, 2), 1.0)]
    @test interactions2M(interactions) == ones(2,2)

    # graph creation
    g = interactions2graph(interactions)
    @test degree(g) == [1,1]
    @test props(g, 1)[:log_energy] == [-1., 1.]
    @test props(g, 2)[:internal_struct] == Dict()
    @test props(g, Edge(1,2))[:J] == 1

    M = [1. 1. 1. 0.; 1. 1. 0. 1.; 1. 0. 1. 1.; 0. 1. 1. 1.]
    interactions = M2interactions(M)

    g1 = interactions2grid_graph(interactions, (2,2))
    println(props(g1, Edge(1,2))[:M])
    println(props(g1, Edge(2,4))[:M])

    interactions = make_interactions()

    #n = Node_of_grid(1, interactions)
    #@test n.connected_nodes == [2,4]

    #n = Node_of_grid(5, interactions)
    #@test n.connected_nodes == [2,6,4,8]

    #n = Node_of_grid(9, interactions)
    #@test n.connected_nodes == [6,8]

    @test get_system_size(interactions) == 9

end

@testset "grid formation" begin
    grid = [1 2 3; 4 5 6; 7 8 9]

    @test is_element(1,1, grid) == true
    @test is_element(1,5, grid) == false
    @test is_element(0,1, grid) == false


    spins_inds = [7 8 9; 10 11 12; 13 14 15]
    i = 2
    v = Element_of_square_grid(i, grid, spins_inds)
    #@test v.i == 2
    @test v.row == 1
    @test v.column == 2
    @test sort(v.spins_inds) == [7,8,9,10,11,12,13,14,15]
    @test v.intra_struct == [(7,8), (8,9), (10,11), (11,12), (13,14), (14,15), (7, 10), (10,13), (8,11), (11,14), (9,12), (12,15)]
    @test v.left == [1,4,7]
    @test v.right == [3,6,9]
    @test v.up == Int[]
    @test v.down == [7,8,9]

    spins_inds = [17 18 19; 20 21 22; 23 24 25]
    i = 5
    v = Element_of_square_grid(i, grid, spins_inds)

    @test v.row == 2
    @test v.column == 2

    @test v.left == [1,4,7]
    @test v.right == [3,6,9]
    @test v.up == [1,2,3]
    @test v.down == [7,8,9]


    spins_inds = [21 25; 22 26; 23 27; 24 28]
    i = 5
    v = Element_of_chimera_grid(i, grid, spins_inds)
    #@test v.i == 5
    @test v.row == 2
    @test v.column == 2
    @test sort(v.spins_inds) == [i for i in 21:28]
    @test length(v.intra_struct) == 16
    #@test v.intra_struct[1:4] == [(21,25), (21,26), (21,27), (21,28)]
    #@test v.left == [5,6,7,8]
    #@test v.right == [5,6,7,8]
    #@test v.up == [1,2,3,4]
    #@test v.down == [1,2,3,4]
end

@testset "axiliary functions for making nodes" begin
    cluster = [2,5,6,7,11]
    @test position_in_cluster(cluster, 6) == 3
    @test position_in_cluster(cluster, 2) == 1

    interactions = make_interactions()
    grid = [1 2;3 4]
    spins_inds = [1 2; 4 5]
    # 1,1 elements of grid with spins [1 2; 4 5]
    v = Element_of_square_grid(1, grid, spins_inds)
    println(log_internal_energy(v, interactions))
end

@testset "node type, connections on the graph" begin

    M = ones(9,9)
    inter = M2interactions(M)

    grid = [1 2 3; 4 5 6; 7 8 9]

    @test nxmgrid(3,3) == grid

    #n = Node_of_grid(3, grid, inter)
    #@test n.i == 3
    #@test n.spins_inds == [3]
    #@test n.intra_struct == Array{Int64,1}[]
    #@test n.left == [1]
    #@test n.right == Int[]
    #@test n.up == Int[]
    #@test n.down == [1]
    #@test n.connected_nodes == [2,6]
    #@test n.connected_spins == [[3 2], [3 6]]

    #n = Node_of_grid(5, grid, inter)
    #@test n.i == 5
    #@test n.spins_inds == [5]
    #@test n.intra_struct == Array{Int64,1}[]
    #@test n.left == [1]
    #@test n.right == [1]
    #@test n.up == [1]
    #@test n.down == [1]
    #@test n.connected_spins == [[5 4], [5 6], [5 2], [5 8]]

    M = ones(16,16)
    inter = M2interactions(M)
    grid1 = [1 2 3 4; 5 6 7 8; 9 10 11 12]
    #n = Node_of_grid(4, grid1, inter)
    #@test n.i == 4
    #@test n.spins_inds == [4]
    #@test n.intra_struct == Array{Int64,1}[]
    #@test n.left == [1]
    #@test n.right == Int[]
    #@test n.up == Int[]
    #@test n.down == [1]


    n = Node_of_grid(9, grid1, inter)
    #@test n.i == 9
    #@test n.spins_inds == [9]
    #@test n.intra_struct == Array{Int64,1}[]
    #@test n.left == Int[]
    #@test n.right == [1]
    #@test n.up == [1]
    #@test n.down == Int[]

    n = Node_of_grid(12, grid1, inter)
    #@test n.i == 12
    #@test n.spins_inds == [12]
    #@test n.intra_struct == Array{Int64,1}[]
    #@test n.left == [1]
    #@test n.right == Int[]
    #@test n.up == [1]
    #@test n.down == Int[]

    #ns = [Node_of_grid(i, grid1, inter) for i in 1:maximum(grid1)]
    #@test get_system_size(ns) == 12

    #grid = Array{Array{Int}}(undef, (2,2))
    #grid[1,1] = [1 2;5 6]
    #grid[1,2] = [3 4; 7 8]
    #grid[2,1] = [9 10;13 14]
    #grid[2,2] = [11 12;15 16]
    #grid = Array{Array{Int}}(grid)

    # chimera

    grid = Array{Array{Int}}(undef, (1,2))
    grid[1,1] = [1 5; 2 6; 3 7; 4 8]
    grid[1,2] = [9 13; 10 14; 11 15; 12 16]
    grid = Array{Array{Int}}(grid)
    M = reshape([1 2], (1,2))

    nc_l = Node_of_grid(1,M, grid, inter; chimera = true)

    #@test nc_l.intra_struct[1:4] == [[1, 5], [1, 6], [1, 7], [1, 8]]
    @test nc_l.spins_inds == [1, 5, 2, 6, 3, 7, 4, 8]
    @test nc_l.right == [2,4,6,8]
    @test nc_l.left == Int[]
    #@test nc_l.left_J == Float64[]

    #@test nc_l.connected_spins[1][:,1] == [5, 6, 7, 8]
    #@test nc_l.connected_spins[1][:,2] == [13, 14, 15, 16]

    #nc_l = Node_of_grid(2,M, grid, inter; chimera = true)
    #@test nc_l.left_J == [1. ,1., 1. ,1.]


    @test chimera_cell(1,1,512) == [1 5; 2 6; 3 7; 4 8]
    @test chimera_cell(1,2,512) == [9 13; 10 14; 11 15; 12 16]
    @test chimera_cell(2,1,512) == [65 69; 66 70; 67 71; 68 72]
end

@testset "test notation" begin

    @testset "Interaction type" begin
        el = Interaction((1,2), 1.1)
        @test el.ind == (1,2)
        @test el.coupling == 1.1

        el = Interaction{BigFloat}((1,2), 1.1)
        @test el.coupling == 1.1
        @test typeof(el.coupling) == BigFloat

        interactions = make_interactions()

        @test getJ(interactions, 1,2) == -0.5
        @test getJ(interactions, 2,1) == -0.5
        @test_throws BoundsError getJ(interactions, 1,3)
    end

    @testset "axiliary" begin
        A = ones(2,2,2,2)
        @test sum_over_last(A) == 2*ones(2,2,2)

        @test last_m_els([1,2,3,4], 2) == [3,4]
        @test last_m_els([1,2,3,4], 5) == [1,2,3,4]

        @test ind2spin(1,1) == [-1]
        @test ind2spin(2,1) == [1]
        @test ind2spin(1, 4) == [-1,-1,-1,-1]
        @test ind2spin(2, 4) == [1,-1,-1,-1]
        @test ind2spin(3, 4) == [-1,1,-1,-1]

        @test spins2ind([-1,-1,-1,-1]) == 1
        @test spins2ind([1,-1,-1,-1]) == 2
        @test spins2ind([-1,1,-1,-1]) == 3

        no_spins = 4
        s = ind2spin(7, no_spins)
        @test s == [-1, 1, 1, -1]
        @test spins2ind(s) == 7

        s = ind2spin(9, no_spins)
        @test s == [-1, -1, -1, 1]
        @test spins2ind(s) == 9


        no_spins = 8
        index_all_spins = 3
        indexes_of_subset = [2,3]

        @test reindex(index_all_spins, no_spins, indexes_of_subset) == 2
        index_all_spins = 1
        @test reindex(index_all_spins, no_spins, indexes_of_subset) == 1


        @test spins2binary([1,1,-1]) == [1,1,0]
        @test binary2spins([0,0,1]) == [-1,-1,1]

        vecvec = [[1,1,-1],[1,-1,1]]
        @test vecvec2matrix(vecvec) == [1 1; 1 -1; -1 1]
    end
end

@testset "brute force testing" begin
    M = ones(3,3)

    v = [1,1,1]
    @test -v2energy(M,v) == -9.

    v = [-1,-1,-1]
    @test -v2energy(M,v) == -3.

    spins, energies = brute_force_solve(M, 2)
    @test spins == [[1, 1, 1], [-1, -1, -1]]
    @test energies == [-9.0, -3.0]

    # first 2 spins must be 1 and others are tho noice
    R = rand(9,9)
    M = 0.01*R*transpose(R)
    M[1,1] = M[1,2] = M[2,1] = M[2,2] = 2
    spins, _ = brute_force_solve(M, 10)
    for i in 1:10
        @test spins[i][1:2] == [1,1]
    end

end
