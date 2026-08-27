@testset "Periodic lattice preserves rectangular dimensions and multiplicity" begin
    m, n, t = 2, 3, 4
    lattice = periodic_lattice((m, n, t))

    @test length(lattice) == 4 * m * n * t
    @test Set(values(lattice)) == Set((i, j) for i in 1:m for j in 1:n)
    @test all(count(==(coord), values(lattice)) == 4 * t for coord in unique(values(lattice)))
    @test extrema(first.(values(lattice))) == (1, m)
    @test extrema(last.(values(lattice))) == (1, n)

    @test length(periodic_lattice((m, n, 2 * t))) == 2 * length(lattice)
end
