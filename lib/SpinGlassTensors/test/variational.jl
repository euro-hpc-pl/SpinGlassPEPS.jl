l = 2
D1 = ([1, 1], [1, 2], [1, 1], [1, 1])
D2 = ([1, 1], [1, 2], [1, 1], [1, 2])
S = Float64
rand_central = rand(CentralTensor{S}, [1, 1, 1, 1, 1, 1, 1, 1])
map1 = MpoTensor(
    TensorMap{S}(
        Dict(
            -1 // 2 => rand_central,
            0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D1),
        ),
    ),
)
map2 = MpoTensor(
    TensorMap{S}(
        Dict(
            -1 // 2 => rand_central,
            0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D2),
        ),
    ),
)
map3 = MpoTensor(
    TensorMap{S}(
        Dict(
            -1 // 2 => rand_central,
            0 => rand(SiteTensor{S}, PoolOfProjectors{Integer}(), l, D1),
        ),
    ),
)
mpomap = Dict(1 => map1, 2 => map2, 3 => map3)

D = 2
sites = [1, 2, 3]
d = [1, 1, 1]
id = Dict(j => d[i] for (i, j) in enumerate(sites))

@testset "Random QMpo with varying physical dimension" begin
    W = rand(QMpo{S}, mpomap)

    @test length(W) == 3
    @test bond_dimension(W) == 1
end

@testset "Compressions for sparse mps and mpo works" begin
    W = rand(QMpo{S}, mpomap)
    ψ = rand(QMps{S}, id, D)
    canonise!(ψ, :left)
    ϕ = rand(QMps{S}, id, D)
    canonise!(ϕ, :left)


end
