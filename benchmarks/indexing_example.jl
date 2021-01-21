using SpinGlassPEPS
using MetaGraphs
using LightGraphs
using Test
using TensorCast

@testset "test weather the solution of the tensor comply with the brute force" begin

    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3 -- 4
    #   |    |  |  |    |
    #   |    |  |  |    |
    #   5 -- 6 -|- 7 -- 8
    #           |

    D = Dict{Tuple{Int64,Int64},Float64}()
    push!(D, (1,1) => 2.5)
    push!(D, (5,5) => 1.4)
    push!(D, (2,2) => 2.3)
    push!(D, (6,6) => 1.2)
    push!(D, (3,3) => -2.5)
    push!(D, (7,7) => -.5)
    push!(D, (4,4) => -.3)
    push!(D, (8,8) => -.2)

    push!(D, (1,5) => 1.3)
    push!(D, (2,6) => -1.)
    push!(D, (3,7) => 1.1)
    push!(D, (4,8) => .1)

    push!(D, (1,2) => .8)
    push!(D, (2,3) => .5)
    push!(D, (3,4) => -1.)

    push!(D, (5,6) => 1.7)
    push!(D, (6,7) => -1.5)
    push!(D, (7,8) => 1.2)

    m = 1
    n = 2
    t = 4

    L = m * n * t

    g_ising = ising_graph(D, L)

    update_cells!(
      g_ising,
      rule = square_lattice((m, 1, n, 1, t)),
    )

    fg = factor_graph(
        g_ising,
        energy=energy,
        spectrum=full_spectrum,
    )


    #println([get_prop(fg, e, :edge) for e in edges(fg)])

    origin = :NW
    β = 2.

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)
    println(pp)

    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]

    sol_A1 = states[[1,2,5,6]]
    sol_A2 = states[[3,4,7,8]]


    # solutions from A1
    # index 3 (right) and % (physical are not trivial)
    A1 = pp[1][1,1,:,1,:]

    # A2 traced
    # index 1 (left is not trivial)
    A2 = MPO(pp)[2][:,1,1,1]

    # contraction of A1 with A2
    #
    #              .           .
    #            .           .
    #   A1 -- A2      =  A12
    #
    A12 = (transpose(A2)*A1)[1,:]

    # maximal margianl probability

    _, spins = findmax(A12)

    st = get_prop(fg, 1, :spectrum).states

    # reading solution from energy numbering and comparison with brute force

    @test st[spins] == sol_A1

    a, b, c = get_prop(fg, 1, 2, :split)
    println(spins)

    println(a)
    println(transpose(c))

    if has_edge(fg, 1, 2)
        p1, en, p2 = get_prop(fg, 1, 2, :split)
    elseif has_edge(fg, 2, 1)
        p2, en, p1 = get_prop(fg, 2, 1, :split)
    else
        p1 = ones(1,1)
        en = p1
    end

    println(size(p1))

    # should be 1 at 2'nd position and is on 1'st
    println(p1[spins, :])
    T = pp[2]

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]
    println(C[1, 1, 1, :])
    #println(size(C))

    _, s = findmax(C[1,1,1,:])

    A2 = pp[2][1, 1, 1, 1, :]
    A2p = pp[2][2, 1, 1, 1, :]
    println(A2)
    println(A2p)

    _, spins_p = findmax(A2p)

    st = get_prop(fg, 2, :spectrum).states

    @test st[spins_p] == sol_A2
    @test st[s] == sol_A2
end
