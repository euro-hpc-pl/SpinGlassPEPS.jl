
@testset "peps_indices correctly assigns indices" begin
m = 3
n = 4

origin_l = [:NW, :NE, :SE, :SW]
origin_r = [:WN, :EN, :ES, :WS]

for (ol, or) ∈ zip(origin_l, origin_r)
    ind_l, i_max_l, j_max_l = peps_indices(m, n, ol)
    ind_r, i_max_r, j_max_r = peps_indices(m, n, or)

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
T = Float64

instance = "$(@__DIR__)/instances/pathological/test_$(m)_$(n)_$(t).txt"

ig = ising_graph(instance)


fg = factor_graph(
    ig,
    energy=energy,
    spectrum=full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

x, y = m, n

for origin ∈ (:NW, :SW, :WS, :WN, :NE, :EN, :SE, :ES)
    peps = PEPSNetwork(x, y, fg, β, origin)

    ψ = IdentityMPS()
    for i ∈ peps.i_max:-1:1
        ψ = MPO(T, peps, i) * ψ
        @test MPS(peps, i) ≈ ψ
    end
end

end
