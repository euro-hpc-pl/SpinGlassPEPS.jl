using SpinGlassPEPS
using MetaGraphs
using LightGraphs
using Test
using TensorCast

@testset "test weather the solution of the tensor comply with the brute force" begin

    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3


    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 3 -|- 5 -- 7
    #   |    |  |  |    |
    #   |    |  |  |    |
    #   2 -- 4 -|- 6 -- 8
    #           |

    D1 = Dict{Tuple{Int64,Int64},Float64}()

    push!(D1, (1,1) => 0.704)
    push!(D1, (2,2) => 0.868)
    push!(D1, (3,3) => 0.592)
    #push!(D1, (4,4) => 0.130)

    push!(D1, (1, 2) => 0.652)
    push!(D1, (2, 3) => 0.730)
    #push!(D1, (3, 4) => 0.314)

    println(D1)
   D = Dict{Tuple{Int64,Int64},Float64}()

   push!(D, (1,1) => 2.5)
   push!(D, (2,2) => 1.4)
   push!(D, (3,3) => 2.3)
   push!(D, (4,4) => 1.2)
   push!(D, (5,5) => -2.5)
   push!(D, (6,6) => -.5)
   push!(D, (7,7) => -.3)
   push!(D, (8,8) => -.2)

   push!(D, (1,2) => 1.3)
   push!(D, (3,4) => -1.)
   push!(D, (5,6) => 1.1)
   push!(D, (7,8) => .1)

   push!(D, (1,3) => .8)
   push!(D, (3,5) => .5)
   push!(D, (5,7) => -1.)

   push!(D, (2,4) => 1.7)
   push!(D, (4,6) => -1.5)
   push!(D, (6,8) => 1.2)

    m = 1
    n = 2
    t = 2

    D = D1

    L = m * n * t

    g_ising = ising_graph(D, L)

    update_cells!(
      g_ising,
      rule = square_lattice((m, 1, n, 1, t)),
    )

    fg = factor_graph(
        g_ising,
        energy=energy,
        spectrum = x -> brute_force(x, num_states=4),
    )

    fg2 = factor_graph(
        g_ising,
        energy=energy,
        spectrum = full_spectrum,
    )


    p1, en, p2 = get_prop(fg, 1, 2, :split)
    r1, sn, r2 = get_prop(fg2, 1, 2, :split)

    #@test size(p1) == size(r1)
    #@test size(p2) == size(r2)

    #@test p1 ≈ r1
    #@test p2 ≈ r2

    println()
    println("brute force projs")
    display(p1)
    println()
    println("energy")
    display(en)
    println()
    println(get_prop(fg, 1, :spectrum))
    println("full spectrum projs")
    display(r1)
    println()
    println("energy")
    display(sn)
    println()
    println(get_prop(fg2, 1, :spectrum))
    println("brute force projs")
    display(p2)
    println()
    println(get_prop(fg, 2, :spectrum))
    println("full spectrum projs")
    display(r2)
    println()
    println(get_prop(fg2, 2, :spectrum))


    if true
        println("vertices of factor graph ", collect(vertices(fg)))
        for v in vertices(fg)
            println("vertex = ", v)
            c = get_prop(fg, v, :cluster)
            println(c.tag)
            println(c.vertices)
            println(c.edges)
            println(c.rank)
            println(c.J)
            println(c.h)
        end

        println("edges of factor graph")
        for e in edges(fg)
            println("edge = ", e)
            println(get_prop(fg, e, :edge))

            p1, e, p2 = get_prop(fg, e, :split)

        end
    end

    D = get_prop(fg, 1, :cluster).vertices
    println("vertices of edge 1")
    vert = keys(D)
    @test 1 in vert
    @test 2 in vert
    #@test 3 in vert
    #@test 4 in vert


    origin = :NW
    β = 2.

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)
    println(pp)


    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]

    println("brute force solution = ", states)
    println()
    display(pp[2][:,1,1,1,:])
    println()

    #sol_A1 = states[[1,2,3,4]]
    #sol_A2 = states[[5,6,7,8]]

    sol_A1 = states[[1,2]]
    sol_A2 = states[[3]]


    # solutions from A1
    # index 3 (right) and % (physical are not trivial)

    Aa1 = pp[1]

    # A2 traced
    # index 1 (left is not trivial)

    Aa2 = MPO(pp)[2]

    # contraction of A1 with A2
    #
    #              .           .
    #            .           .
    #   A1 -- A2      =  A12
    #
    #A12 = (transpose(A2)*A1)[1,:]
    @reduce A12[l, u, d, uu, rr, dd, σ] |= sum(x) Aa1[l, u, x, d, σ] * Aa2[x, uu, rr, dd]
    #@test size(A12) == (1,1,1,1,1,1,16)
    A12 = dropdims(A12, dims=(1,2,3,4,5,6))


    # maximal margianl probability


    _, spins = findmax(A12)

    st = get_prop(fg, 1, :spectrum).states

    # reading solution from energy numbering and comparison with brute force

    @test st[spins] == sol_A1

    #println(spins)

    if has_edge(fg, 1, 2)
        p1, en, p2 = get_prop(fg, 1, 2, :split)
    elseif has_edge(fg, 2, 1)
        p2, en, p1 = get_prop(fg, 2, 1, :split)
    else
        p1 = ones(1,1)
        en = p1
    end

    # should be 1 at 2'nd position and is on 1'st
    println(spins)
    println(p1[spins, :])
    T = pp[2]

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]

    println()
    display(C[1,1,1,:])
    println()

    _, s = findmax(C[1,1,1,:])

    st = get_prop(fg, 2, :spectrum).states
    println(st)
    @test st[s] == sol_A2

    A2p = pp[2][1, 1, 1, 1, :]
    _, spins_p = findmax(A2p)

    println(st[spins_p] == sol_A2)

end
