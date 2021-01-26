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


    D1 = Dict((1, 2) => 0.652,(2, 3) => 0.73,(3, 3) => 0.592,(2, 2) => 0.704,(1, 1) => 0.868)

    D2 = Dict((1, 2) => -0.9049,(2, 3) => 0.2838,(3, 3) => -0.7928,(2, 2) => 0.1208,(1, 1) => -0.3342)

    for D in [D1, D2]

        m = 1
        n = 2
        t = 2

        L = m * n * t

        g_ising = ising_graph(D, L)

        # brute force solution
        bf = brute_force(g_ising; num_states = 1)
        states = bf.states[1]
        sol_A1 = states[[1,2]]
        sol_A2 = states[[3]]

        #particular form of peps tensors
        update_cells!(
          g_ising,
          rule = square_lattice((m, 1, n, 1, t)),
        )

        fg = factor_graph(
            g_ising,
            Dict(1=>4, 2=>2),
            energy=energy,
            spectrum = brute_force,
        )

        origin = :NW
        β = 1.

        x, y = m, n
        peps = PepsNetwork(x, y, fg, β, origin)
        pp = PEPSRow(peps, 1)
        if D[(1, 2)] == 0.652
            h1 = D[(1,1)]
            h2 = D[(2,2)]
            J12 = D[(1,2)]
            J23 = D[(2,3)]
            h3 = D[(3,3)]

            println("matricised A1")
            display(pp[1][1,1,:,1,:])
            println()
            #its explicite form including J23
            A1ex = [exp(h1+h2-J12-J23) exp(h1-h2+J12+J23) exp(-h1+h2+J12-J23) exp(-h1-h2-J12+J23); exp(h1+h2-J12+J23) exp(h1-h2+J12-J23) exp(-h1+h2+J12+J23) exp(-h1-h2-J12-J23)]
            println("A1 equal to its explicite form including J23")
            println(pp[1][1,1,:,1,:] ≈ A1ex)
            # but J23 should be in A2

            println("matricised A2")
            display(pp[2][:,1,1,1,:])
            println()
            println("its explicite form do not include J23")
            #but should
            println(pp[2][:,1,1,1,:] ≈ [exp(h3) 0.; 0. exp(-h3)])
            println("why A2 is diagonal")
        end





        # peps solution
        Aa1 = pp[1]
        Aa2 = MPO(pp)[2]
        @reduce A12[l, u, d, uu, rr, dd, σ] |= sum(x) Aa1[l, u, x, d, σ] * Aa2[x, uu, rr, dd]
        A12 = dropdims(A12, dims=(1,2,3,4,5,6))
        _, spins = findmax(A12)

        #solution from the first tensor
        st = get_prop(fg, 1, :spectrum).states
        @test st[spins] == sol_A1

        if has_edge(fg, 1, 2)
            p1, en, p2 = get_prop(fg, 1, 2, :split)
        elseif has_edge(fg, 2, 1)
            p2, en, p1 = get_prop(fg, 2, 1, :split)
        else
            p1 = ones(1,1)
            en = p1
        end
        T = pp[2]
        @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]
        _, s = findmax(C[1,1,1,:])

        # solution form the second tensor
        st = get_prop(fg, 2, :spectrum).states
        @test st[s] == sol_A2
    end
end