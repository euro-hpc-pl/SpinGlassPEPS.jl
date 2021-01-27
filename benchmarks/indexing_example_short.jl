using SpinGlassPEPS
using MetaGraphs
using LightGraphs
using Test
using TensorCast

function read_projectors(fg, i::Int, j::Int)
    if has_edge(fg, i, j)
        p1, en, p2 = get_prop(fg, i, j, :split)
    elseif has_edge(fg, j, i)
        p2, en, p1 = get_prop(fg, j, i, :split)
    else
        p1 = ones(1,1)
        en = p1
        p2 = p1
    end
    p1, en, p2
end

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
            spectrum = brute_force
        )

        origin = :NW
        β = 2.

        x, y = m, n
        peps = PepsNetwork(x, y, fg, β, origin)
        pp = PEPSRow(peps, 1)

        # the solution without cutting off
        M1 = pp[1][1,1,:,1,:]
        M2 = pp[2][:,1,1,1,:]
        @reduce MM[a,b] |= sum(x) M1[x,a]*M2[x,b]

        _, inds = findmax(MM)

        h1 = D[(1,1)]
        h2 = D[(2,2)]
        J12 = D[(1,2)]
        J23 = D[(2,3)]
        h3 = D[(3,3)]

        if D[(1, 2)] == 0.652
            A1ex = reshape([exp(h1+h2-J12) 0. exp(-h1+h2+J12) 0.; 0. exp(h1-h2+J12) 0. exp(-h1-h2-J12)], (1,1,2,1,4))
            A2ex = reshape([exp(h3-J23) exp(-h3+J23); exp(h3+J23) exp(-h3-J23)],(2,1,1,1,2))
            C = [exp(h1+h2-J12+h3-J23) exp(h1-h2+J12+h3+J23) exp(-h1+h2+J12+h3-J23) exp(-h1-h2-J12+h3+J23); exp(h1+h2-J12-h3+J23) exp(h1-h2+J12-h3-J23) exp(-h1+h2+J12-h3+J23) exp(-h1-h2-J12-h3-J23)]
        else
            A1ex = reshape([exp(-h1-h2-J12) 0. 0. exp(h1-h2+J12); 0. exp(h1+h2-J12) exp(-h1+h2+J12) 0.], (1,1,2,1,4))
            A2ex = reshape([exp(-h3-J23) exp(h3+J23); exp(-h3+J23) exp(h3-J23)],(2,1,1,1,2))
            C = [exp(-h1-h2-J12-h3-J23) exp(h1+h2-J12-h3+J23) exp(-h1+h2+J12-h3+J23) exp(h1-h2+J12-h3-J23); exp(-h1-h2-J12+h3+J23) exp(h1+h2-J12+h3-J23) exp(-h1+h2+J12+h3-J23) exp(h1-h2+J12+h3+J23)]

            println("states of T1  ", get_prop(fg, 1, :spectrum).states)
            println("x")
            println("states of T2  ", get_prop(fg, 2, :spectrum).states)
            println("D2, tensor with beta = 1")
            display(C)
            println()

            println("D2, tensor with beta = $β")
            display(C.^β)
            println()
        end

        @test pp[1] ≈ A1ex.^β
        @test pp[2] ≈ A2ex.^β
        @test MM ≈ transpose(C.^β)

        # peps solution, first tensor
        Aa1 = pp[1]
        Aa2 = MPO(pp)[2]

        @reduce A12[l, u, d, uu, rr, dd, σ] |= sum(x) Aa1[l, u, x, d, σ] * Aa2[x, uu, rr, dd]
        A12 = dropdims(A12, dims=(1,2,3,4,5,6))
        _, spins = findmax(A12)

        #solution from the first tensor
        st = get_prop(fg, 1, :spectrum).states
        @test st[spins] == sol_A1
        @test st[inds[1]] == sol_A1

        # reading projector
        p1, en, p2 = read_projectors(fg, 1, 2)
        if D[(1, 2)] == 0.652
            @test p1 == [1.0 0.0; 0.0 1.0; 1.0 0.0; 0.0 1.0]
            @test en == [0.73 -0.73; -0.73 0.73]
            @test p2 == [1.0 0.0; 0.0 1.0]
        end
        r1, rn, r2 = read_projectors(fg, 2, 1)
        @test p1 == r2
        @test p2 == r1
        @test en == rn
        @test read_projectors(fg, 3, 1) == (ones(1,1), ones(1,1), ones(1,1))

        @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * pp[$2][x, a, b, c, d]
        _, s = findmax(C[1,1,1,:])

        # solution form the second tensor
        st = get_prop(fg, 2, :spectrum).states
        @test st[s] == sol_A2
        @test st[inds[2]] == sol_A2
    end
end

@testset "larger example" begin
    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 3 -|- 5 -- 7
    #   |    |  |  |    |
    #   |    |  |  |    |
    #   2 -- 4 -|- 6 -- 8
    #           |

   D = Dict((5, 7) => -0.0186,(5, 6) => 0.0322,(2, 2) => -0.5289544745642463,(4, 4) => -0.699,(4, 6) => 0.494,(3, 3) => -0.4153941108520631,(8, 8) => 0.696,(6, 8) => 0.552,(1, 3) => -0.739,(7, 8) => -0.0602,(2, 4) => -0.0363,(1, 1) => 0.218,(7, 7) => -0.931,(1, 2) => 0.162,(6, 6) => 0.567,(5, 5) => -0.936,(3, 4) => 0.0595,(3, 5) => -0.9339)


    m = 1
    n = 2
    t = 4


    L = m * n * t

    g_ising = ising_graph(D, L)

    update_cells!(
      g_ising,
      rule = square_lattice((m, 1, n, 1, t)),
    )

    fg1 = factor_graph(
        g_ising,
        Dict(1=>16, 2=>16),
        energy=energy,
        spectrum = brute_force,
    )

    fg2 = factor_graph(
        g_ising,
        Dict(1=>16, 2=>16),
        energy=energy,
        spectrum = full_spectrum,
    )

    for fg in [fg1, fg2]

        #Partition function
        β = 2.
        states = collect.(all_states(rank_vec(g_ising)))

        ρ = exp.(-β .* energy.(states, Ref(g_ising)))
        Z = sum(ρ)
        @test gibbs_tensor(g_ising, β)  ≈ ρ ./ Z

        for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)

            peps = PepsNetwork(m, n, fg, β, origin)

            ψ = MPO(PEPSRow(peps, 1))

            for i ∈ 2:peps.i_max
                W = MPO(PEPSRow(peps, i))
                M = MPO(peps, i-1, i)

                ψ = (ψ * M) * W

                @test length(W) == peps.j_max
                for A ∈ ψ @test size(A, 2) == 1 end
                @test size(ψ[1], 1) == 1 == size(ψ[peps.j_max], 3)
            end
            for A ∈ ψ @test size(A, 4) == 1 end
            #println("ψ ", ψ)

            ZZ = []
            for A ∈ ψ
                #println("A ", A)
                push!(ZZ, dropdims(A, dims=(2, 4)))
                #println("ZZ ", ZZ)
            end
            @test Z ≈ prod(ZZ)[]
        end


        origin = :NW

        x, y = m, n
        peps = PepsNetwork(x, y, fg, β, origin)
        pp = PEPSRow(peps, 1)


        # brute force solution
        bf = brute_force(g_ising; num_states = 1)
        states = bf.states[1]

        cluster = props(fg, 1)[:cluster]
        println(cluster.vertices)

        cluster = props(fg, 2)[:cluster]
        println(cluster.vertices)
        sol_A1 = states[[1,2,3,4]]
        sol_A2 = states[[5,6,7,8]]

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

        @reduce A12[l, u, d, uu, rr, dd, σ] |= sum(x) Aa1[l, u, x, d, σ] * Aa2[x, uu, rr, dd]

        A12 = dropdims(A12, dims=(1,2,3,4,5,6))

        _, spins = findmax(A12)

        st = get_prop(fg, 1, :spectrum).states

        @test st[spins] == sol_A1

        p1, en, p2 = read_projectors(fg, 1, 2)

        @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * pp[$2][x, a, b, c, d]

        _, s = findmax(C[1,1,1,:])

        st = get_prop(fg, 2, :spectrum).states

        @test st[s] == sol_A2
    end
end
