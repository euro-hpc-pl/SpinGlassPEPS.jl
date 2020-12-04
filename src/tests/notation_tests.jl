@testset "grids" begin

    # grid
    grid = [1 2 3; 4 5 6; 7 8 9]
    @test nxmgrid(3,3) == grid
    @test nxmgrid(2,4) == [1 2 3 4; 5 6 7 8]

    c = grid_cel(1,1,(2,2),(5,5))
    @test c == [1 2; 6 7]
    c = grid_cel(1,3,(2,2),(5,5))
    @test c == reshape([5; 10], (2,1))
    c = grid_cel(3,3,(2,2),(5,5))
    @test c == reshape([25], (1,1))

    grid, M = form_a_grid((2,3), (7,7))
    @test M == [1 2 3; 4 5 6; 7 8 9; 10 11 12]
    @test grid[1,1] == [1 2 3; 8 9 10]
    @test grid[4,1] == reshape([43 44 45], (1,3))
    @test grid[4,3] == reshape([49], (1,1))

    # chimera

    @test chimera_cell(1,1,512) == [1 5; 2 6; 3 7; 4 8]
    @test chimera_cell(1,2,512) == [9 13; 10 14; 11 15; 12 16]
    @test chimera_cell(2,1,512) == [65 69; 66 70; 67 71; 68 72]


    grid, M = form_a_chimera_grid(2, (1,1))

    @test grid[1,1] == [1 5; 2 6; 3 7; 4 8]
    @test grid[1,2] == [9 13; 10 14; 11 15; 12 16]
    @test M == [1 2; 3 4]

    grid, M = form_a_chimera_grid(2, (1,2))
    println(grid)
    println(M)

    grid, M = form_a_chimera_grid(2, (2,1))
    println(grid)
    println(M)


    grid, M = form_a_chimera_grid(2, (2,2))
    println(grid)
    println(M)

    M = ones(4,4)
    fullM2grid!(M, (2,2))
    @test M == [1.0 1.0 1.0 0.0; 1.0 1.0 0.0 1.0; 1.0 0.0 1.0 1.0; 0.0 1.0 1.0 1.0]
end


@testset "grid formation" begin
    # grid
    cluster = [2,5,6,7,11]
    @test position_in_cluster(cluster, 6) == 3
    @test position_in_cluster(cluster, 2) == 1
    @test positions_in_cluster(cluster, [11,6]) == [5,3]

    grid = [1 2 3; 4 5 6; 7 8 9]
    spins_inds = [7 8 9; 10 11 12; 13 14 15]
    i = 2
    v = Element_of_square_grid(i, grid, spins_inds)

    @test v.row == 1
    @test v.column == 2
    @test sort(v.spins_inds) == [7,8,9,10,11,12,13,14,15]
    #@test v.intra_struct == [(7,8), (8,9), (10,11), (11,12), (13,14), (14,15), (7, 10), (10,13), (8,11), (11,14), (9,12), (12,15)]
    @test v.left == [1,2,3]
    @test v.right == [7,8,9]
    @test v.up == Int[]
    @test v.down == [3,6,9]

    i = 5
    v = Element_of_square_grid(i, grid)

    @test v.row == 2
    @test v.column == 2
    #@test v.intra_struct == []
    @test v.spins_inds == [5]
    @test v.left == [1]
    @test v.right == [1]
    @test v.up == [1]
    @test v.down == [1]

    M = Array{Union{Nothing, Array{Int}}}(nothing, (1,1))
    M[1,1] = [1 2; 3 4]
    M = Matrix{Array{Int, N} where N}(M)
    grid = reshape([1], (1,1))

    v = Element_of_square_grid(1, grid, M)
    @test v.row == 1
    @test v.column == 1


    # chimera
    grid = [1 2 3; 4 5 6; 7 8 9]
    spins_inds = [21 25; 22 26; 23 27; 24 28]
    i = 5
    v = Element_of_chimera_grid(i, grid, spins_inds)
    @test v.row == 2
    @test v.column == 2
    @test sort(v.spins_inds) == [i for i in 21:28]
    #@test length(v.intra_struct) == 16
    #@test v.intra_struct[1:4] == [(21,25), (21,26), (21,27), (21,28)]

    @test v.left == [5, 6, 7, 8]
    @test v.right == [5, 6, 7, 8]
    @test v.up == [1,2,3,4]
    @test v.down == [1,2,3,4]
end


@testset "graph representation" begin
    M = ones(2,2)

    # graph for mps
    g = M2graph(M)
    @test collect(vertices(g)) == [1,2]

    @test props(g, 1)[:h] == 1.
    @test props(g, 2)[:h] == 1.
    @test props(g, 1,2)[:J] == 2.

    g1 = graph4mps(g)
    @test degree(g1) == [1,1]
    @test props(g1, 1)[:energy] == [1., -1.]
    @test props(g1, 1,2)[:J] == 2.


    # graph for peps
    M = ones(4,4)
    fullM2grid!(M, (2,2))
    ig = M2graph(M)

    g1 = graph4peps(ig, (1,1))
    @test props(g1, 1,2)[:M] == [-2.0 2.0; 2.0 -2.0]
    @test props(g1, 2,4)[:M] == [-2.0 2.0; 2.0 -2.0]
    @test props(g1, 1)[:energy] == [1., -1.]
    @test props(g1, 2)[:energy] == [1., -1.]
    @test props(g1, 1,2)[:inds] == [1]

    M = ones(16,16)
    fullM2grid!(M, (4,4))
    ig = M2graph(M)
    g1 = graph4peps(ig, (2,2))

    M = props(g1, 1,2)[:M]
    @test size(M) == (4,16)
    @test M[:,1] == [-4.0, 0.0, 0.0, 4.0]
    e = [-4.0, 2.0, 2.0, 0.0, 2.0, 0.0, 8.0, -2.0, 2.0, 8.0, 0.0, -2.0, 0.0, -2.0, -2.0, -12.0]
    @test props(g1, 1)[:energy] == e
    @test props(g1, 1,2)[:inds] == [3, 4]
    @test props(g1, 1,3)[:inds] == [2, 4]

end


@testset "energy computation for a node" begin

    M = ones(16,16)
    fullM2grid!(M, (4,4))
    ig = M2graph(M)

    grid = [1 2;3 4]
    spins_inds = [1 2; 5 6]
    # 1,1 elements of grid with spins [1 2; 4 5]
    v1 = Element_of_square_grid(1, grid, spins_inds)
    spins_inds1 = [3 4; 7 8]
    v2 = Element_of_square_grid(2, grid, spins_inds1)

    e =  [-4.0, 2.0, 2.0, 0.0, 2.0, 0.0, 8.0, -2.0, 2.0, 8.0, 0.0, -2.0, 0.0, -2.0, -2.0, -12.0]
    #@test internal_energy(v1, ig) ≈ e
    @test get_Js(v2, v1, ig) == [2.0, 2.0]
    M = M_of_interaction(v2, v1, ig)
    @test size(M) == (4, 16)
    @test M[:,1] == [-4.0, 0.0, 0.0, 4.0]
end

@testset "operations on spins" begin

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

    @test spins2binary([-1,-1,1,1]) == [0,0,1,1]
    @test spins2binary([1,1,-1]) == [1,1,0]
    @test binary2spins([0,0,1]) == [-1,-1,1]

    A = ones(2,2,2,2)
    @test sum_over_last(A) == 2*ones(2,2,2)
    @test last_m_els([1,2,3,4], 2) == [3,4]
    @test last_m_els([1,2,3,4], 5) == [1,2,3,4]

    M = ones(16,16)
    fullM2grid!(M, (4,4))
    ig = M2graph(M)
    g1 = graph4peps(ig, (2,2))
    @test get_system_size(g1) == 16

    vecvec = [[1,1,-1],[1,-1,1]]
    @test vecvec2matrix(vecvec) == [1 1; 1 -1; -1 1]
end
