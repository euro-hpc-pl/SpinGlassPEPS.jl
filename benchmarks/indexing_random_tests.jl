using SpinGlassPEPS
using MetaGraphs
using LightGraphs
using Test
using TensorCast

function read_projectors(fg::MetaDiGraph, i::Int, j::Int)
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


f() = 2 * rand() - 1

@testset "random test the solution of the tensor with the brute force, random case" begin

    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3

    D = Dict{Tuple{Int64,Int64},Float64}()


    push!(D, (2,2) => f())
    push!(D, (1,1) => f())
    push!(D, (3,3) => f())

    push!(D, (1, 2) => f())
    push!(D, (2, 3) => f())

    println("small example ", D)

    m = 1
    n = 2
    t = 2

    L = m * n * t

    g_ising = ising_graph(D, L-1)


    update_cells!(
      g_ising,
      rule = square_lattice((m, 1, n, 1, t)),
    )

    fg = factor_graph(
        g_ising,
        Dict(1=>4, 2=>2),
        energy=energy,
        spectrum = brute_force,
        #spectrum = full_spectrum,
    )

    β = 2.

    origin = :NW

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)

    # the solution without cutting off, it works
    M1 = pp[1][1,1,:,1,:]
    M2 = pp[2][:,1,1,1,:]
    @reduce MM[a,b] |= sum(x) M1[x,a]*M2[x,b]
    _, inds = findmax(MM)

    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]


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

    @reduce A12[l, u, d, uu, rr, dd, σ] |= sum(x) Aa1[l, u, x, d, σ] * Aa2[x, uu, rr, dd]
    @test size(A12) == (1,1,1,1,1,1,4)
    A12 = dropdims(A12, dims=(1,2,3,4,5,6))

    # maximal margianl probability

    _, spins = findmax(A12)

    st = get_prop(fg, 1, :spectrum).states

    # reading solution from energy numbering and comparison with brute force

    @test st[spins] == sol_A1
    @test st[inds[1]] == sol_A1

    p1, _, _ = read_projectors(fg, 1, 2)

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * pp[$2][x, a, b, c, d]

    _, s = findmax(C[1,1,1,:])

    st = get_prop(fg, 2, :spectrum).states

    @test st[s] == sol_A2
    @test st[inds[2]] == sol_A2
end


@testset "random larger example" begin
    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 3 -|- 5 -- 7
    #   |    |  |  |    |
    #   |    |  |  |    |
    #   2 -- 4 -|- 6 -- 8
    #           |

   D = Dict((5, 7) => f() ,(5, 6) => f() ,(2, 2) => f() ,(4, 4) => f() ,(4, 6) => f() ,(3, 3) => f() ,(8, 8) => f() ,(6, 8) => f() ,(1, 3) => f() ,(7, 8) => f() ,(2, 4) => f() ,(1, 1) => f() ,(7, 7) => -0.931,(1, 2) => 0.162,(6, 6) => f() ,(5, 5) => f() ,(3, 4) => f() ,(3, 5) => f())

   println("larger example ", D)

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

        #println(spins)

        p1, _, _ = read_projectors(fg, 1, 2)

        @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * pp[$2][x, a, b, c, d]

        _, s = findmax(C[1,1,1,:])

        st = get_prop(fg, 2, :spectrum).states

        @test st[s] == sol_A2
    end
end
