using SpinGlassPEPS
using MetaGraphs
using LightGraphs
using Test
using TensorCast

if false
@testset "test the solution of the tensor with the brute force, random case" begin

    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3

    D = Dict{Tuple{Int64,Int64},Float64}()
    f() = 2*rand()-1

    push!(D, (2,2) => f())
    push!(D, (1,1) => f())
    push!(D, (3,3) => f())


    push!(D, (1, 2) => f())
    push!(D, (2, 3) => f())


    m = 1
    n = 2
    t = 2

    println(D)

    L = m * n * t

    g_ising = ising_graph(D, L)

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

    fg2 = factor_graph(
        g_ising,
        Dict(1=>4, 2=>2),
        energy=energy,
        spectrum = full_spectrum,
    )

    #Partition function
    β = 1
    states = collect.(all_states(get_prop(g_ising, :rank)))
    println("states ", states)
    ρ = exp.(-β .* energy.(states, Ref(g_ising)))
    Z = sum(ρ)
    println("Z ", Z)

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
        println("ψ ", ψ)

        ZZ = []
        for A ∈ ψ
            println("A ", A)
            push!(ZZ, dropdims(A, dims=(2, 4)))
            println("ZZ ", ZZ)
        end
        @test Z ≈ 2*prod(ZZ)[]
    end


    sp = get_prop(fg, 1, :spectrum)

    sp2 = get_prop(fg2, 1, :spectrum)

    println("brute force (local) factor graph")
    display(sp.states)
    println()
    println("fill spectrum factor graph")
    display(sp2.states)
    println()

    p1, en, p2 = get_prop(fg, 1, 2, :split)
    r1, sn, r2 = get_prop(fg2, 1, 2, :split)

    #@test size(p1) == size(r1)
    #@test size(p2) == size(r2)

    #@test p1 ≈ r1
    #@test p2 ≈ r2

    println()
    println("brute force (local) projector")
    display(p1)
    println()
    println("full spectrum (local) projector")
    display(r1)
    println()


    origin = :NW
    β = 1.

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)
    println(pp)

    println("matricised A1")
    display(pp[1][1,1,:,1,:])
    println()

    println("matricised A2")
    display(pp[2][:,1,1,1,:])
    println()




    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]

    println("brute force (global) solution = ", states)

    println("it sould equal to the peps solution that is:")

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

    println("solution of A1, index = ", spins, " partial configuration = ", st[spins])

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
    println("projector again")
    display(p1)
    println()
    println("second projector")
    display(p2)
    println()

    println("this correspond to following row of the projector  = ", p1[spins, :])

    println("and index = ", findall(p1[spins, :] .== 1))

    T = pp[2]

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]


    println("its selected row at index ", findall(p1[spins, :] .== 1))
    display(C[1,1,1,:])
    println()

    _, s = findmax(C[1,1,1,:])

    st = get_prop(fg, 2, :spectrum).states
    println("the solution of A2 is indexed by   ", s)

    println("spectrum of A2 ", st)
    println("it gives the solution is spin configuration ", st[s])
    println("and global brute force gives ", sol_A2)

    @test st[s] == sol_A2

    #arbitrary other index for which it works
    A2p = pp[2][1, 1, 1, 1, :]
    _, spins_p = findmax(A2p)

    println(st[spins_p] == sol_A2)

end
end

if true
@testset "test weather the solution of the tensor comply with the brute force" begin

    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 2 -|- 3

    D = Dict{Tuple{Int64,Int64},Float64}()

    push!(D, (2,2) => 0.704)
    push!(D, (1,1) => 0.868)
    push!(D, (3,3) => 0.592)


    push!(D, (1, 2) => 0.652)
    push!(D, (2, 3) => 0.730)


    m = 1
    n = 2
    t = 2

    L = m * n * t

    g_ising = ising_graph(D, L)

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

    fg2 = factor_graph(
        g_ising,
        Dict(1=>4, 2=>2),
        energy=energy,
        spectrum = full_spectrum,
    )

    #Partition function
    β = 1
    states = collect.(all_states(get_prop(g_ising, :rank)))
    println("states ", states)
    ρ = exp.(-β .* energy.(states, Ref(g_ising)))
    Z = sum(ρ)
    println("Z ", Z)

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
        println("ψ ", ψ)

        ZZ = []
        for A ∈ ψ
            println("A ", A)
            push!(ZZ, dropdims(A, dims=(2, 4)))
            println("ZZ ", ZZ)
        end
        @test Z ≈ 2*prod(ZZ)[]
    end


    sp = get_prop(fg, 1, :spectrum)

    sp2 = get_prop(fg2, 1, :spectrum)

    println("brute force (local) factor graph")
    display(sp.states)
    println()
    println("fill spectrum factor graph")
    display(sp2.states)
    println()

    p1, en, p2 = get_prop(fg, 1, 2, :split)
    r1, sn, r2 = get_prop(fg2, 1, 2, :split)

    #@test size(p1) == size(r1)
    #@test size(p2) == size(r2)

    #@test p1 ≈ r1
    #@test p2 ≈ r2

    println()
    println("brute force (local) projector")
    display(p1)
    println()
    println("full spectrum (local) projector")
    display(r1)
    println()

    #=
    if false
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
    =#


    origin = :NW
    β = 1.

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)
    println(pp)

    println("matricised A1")
    display(pp[1][1,1,:,1,:])
    println()
    h1 = D[(1,1)]
    h2 = D[(2,2)]
    J12 = D[(1,2)]
    J23 = D[(2,3)]
    println(pp[1][1,1,:,1,:] ≈ [exp(h1+h2-J12-J23) exp(h1-h2+J12+J23) exp(-h1+h2+J12-J23) exp(-h1-h2-J12+J23); exp(h1+h2-J12+J23) exp(h1-h2+J12-J23) exp(-h1+h2+J12+J23) exp(-h1-h2-J12-J23)])
    println()

    h3 = D[(3,3)]
    println("matricised A2")
    display(pp[2][:,1,1,1,:])
    println()
    println(pp[2][:,1,1,1,:] ≈ [exp(h3) 0.; 0. exp(-h3)])
    println()
    println("why it is diagonal?")

    # the solution without cutting off, it works
    M1 = pp[1][1,1,:,1,:]
    M2 = pp[2][:,1,1,1,:]
    @reduce MM[a,b] |= sum(x) M1[x,a]*M2[x,b]
    println(MM)
    _, inds = findmax(MM)


    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]

    println("brute force (global) solution = ", states)

    println("it sould equal to the peps solution that is:")

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

    println("solution of A1, index = ", spins, " partial configuration = ", st[spins])

    # reading solution from energy numbering and comparison with brute force

    @test st[spins] == sol_A1
    @test st[inds[1]] == sol_A1

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
    println("projector again")
    display(p1)
    println()
    println("second projector")
    display(p2)
    println()

    println("this correspond to following row of the projector  = ", p1[spins, :])

    println("and index = ", findall(p1[spins, :] .== 1))

    T = pp[2]

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]


    println("its selected row at index ", findall(p1[spins, :] .== 1))
    display(C[1,1,1,:])
    println()

    _, s = findmax(C[1,1,1,:])

    st = get_prop(fg, 2, :spectrum).states
    println("the solution of A2 is indexed by   ", s)

    println("spectrum of A2 ", st)
    println("it gives the solution is spin configuration ", st[s])
    println("and global brute force gives ", sol_A2)

    @test st[s] == sol_A2
    @test st[inds[2]] == sol_A2

    #arbitrary other index for which it works
    A2p = pp[2][1, 1, 1, 1, :]
    _, spins_p = findmax(A2p)

    println(st[spins_p] == sol_A2)

end
end

if false
@testset "lerger example" begin
    #      grid
    #     A1    |    A2
    #           |
    #   1 -- 3 -|- 5 -- 7
    #   |    |  |  |    |
    #   |    |  |  |    |
    #   2 -- 4 -|- 6 -- 8
    #           |

   D = Dict((5, 7) => -0.0186,(5, 6) => 0.0322,(2, 2) => -0.5289544745642463,(4, 4) => -0.699,(4, 6) => 0.494,(3, 3) => -0.4153941108520631,(8, 8) => 0.696,(6, 8) => 0.552,(1, 3) => -0.739,(7, 8) => -0.0602,(2, 4) => -0.0363,(1, 1) => 0.218,(7, 7) => -0.931,(1, 2) => 0.162,(6, 6) => 0.567,(5, 5) => -0.936,(3, 4) => 0.0595,(3, 5) => -0.9339)

   println(D)

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

    #Partition function
    β = 1
    states = collect.(all_states(get_prop(g_ising, :rank)))
    println("states ", states)
    ρ = exp.(-β .* energy.(states, Ref(g_ising)))
    Z = sum(ρ)
    println("Z ", Z)

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
        println("ψ ", ψ)

        ZZ = []
        for A ∈ ψ
            println("A ", A)
            push!(ZZ, dropdims(A, dims=(2, 4)))
            println("ZZ ", ZZ)
        end
        @test Z ≈ prod(ZZ)[]
    end


    origin = :NW
    β = 1.

    x, y = m, n
    peps = PepsNetwork(x, y, fg, β, origin)
    pp = PEPSRow(peps, 1)

    println(pp)


    # brute force solution
    bf = brute_force(g_ising; num_states = 1)
    states = bf.states[1]

    println("brute force (global) solution = ", states)

    println("it sould equal to the peps solution that is:")

    sol_A1 = states[[1,2,3,4]]
    sol_A2 = states[[5,6,7,8]]


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

    A12 = dropdims(A12, dims=(1,2,3,4,5,6))


    _, spins = findmax(A12)

    st = get_prop(fg, 1, :spectrum).states

    println("solution of A1, index = ", spins, " partial configuration = ", st[spins])

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


    println("this correspond to following row of the projector  = ", p1[spins, :])

    println("and index = ", findall(p1[spins, :] .== 1))

    T = pp[2]

    @reduce C[a, b, c, d] := sum(x) p1[$spins, x] * T[x, a, b, c, d]

    println("matricised A2")
    display(pp[2][:,1,1,1,:])
    println()

    println("its selected row at index ", findall(p1[spins, :] .== 1))
    display(C[1,1,1,:])
    println()

    _, s = findmax(C[1,1,1,:])

    st = get_prop(fg, 2, :spectrum).states
    println("the solution of A2 is indexed by   ", s)

    println("spectrum of A2 ", st)
    println("it gives the solution is spin configuration ", st[s])
    println("and global brute force gives ", sol_A2)

    @test st[s] == sol_A2

    #arbitrary other index for which it works
    A2p = pp[2][1, 1, 1, 1, :]
    _, spins_p = findmax(A2p)

    println(st[spins_p] == sol_A2)


end
end
