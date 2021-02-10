
@testset "LinearIndices correctly assigns indices" begin
m = 3
n = 4

origin_l = [:NW, :NE, :SE, :SW]
origin_r = [:WN, :EN, :ES, :WS]

for (ol, or) ∈ zip(origin_l, origin_r)
    ind_l, i_max_l, j_max_l = LinearIndices(m, n, ol)
    ind_r, i_max_r, j_max_r = LinearIndices(m, n, or)

    @test i_max_l == m == j_max_r
    @test j_max_l == n == i_max_r

    for i ∈ 0:m+1, j ∈ 0:n+1
        @test ind_l[i, j] == ind_r[j, i]
    end
end
end

@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1

L = m * n * t

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

ig = ising_graph(instance, L)
update_cells!(
   ig,
   rule = square_lattice((m, n, t)),
)

fg = factor_graph(
    ig,
    energy=energy,
    spectrum=full_spectrum,
)

x, y = m, n

for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)

    peps = PepsNetwork(x, y, fg, β, origin)
    @test typeof(peps) == PepsNetwork

    ψ = idMPS(peps.j_max)
    ψ_all = boundaryMPS(peps)

    for i ∈ peps.i_max:-1:1
        ψ = MPO(eltype(ψ), peps, i) * ψ
        @test ψ_all[i] ≈ ψ
    end
end

end
