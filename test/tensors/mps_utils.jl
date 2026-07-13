function simple_dense_mpo(::Type{T}, sites) where {T}
    tensors = MpoTensorMap{T}()
    for site ∈ sites
        core = zeros(T, 1, 2, 1, 2)
        core[1, 1, 1, 1] = one(T)
        core[1, 2, 1, 2] = one(T)
        tensors[site] = MpoTensor(TensorMap{T}(0 => core))
    end
    QMpo(tensors)
end

@testset "QMps and QMpo collection size" begin
    sites = Site[1 // 2, 2, 7 // 2]
    tensors = TensorMap{Float64}(
        sites[1] => randn(1, 2, 2),
        sites[2] => randn(2, 2, 3),
        sites[3] => randn(2, 1, 2),
    )
    ψ = QMps(tensors)
    W = simple_dense_mpo(Float64, sites)

    @test length(ψ) == 3
    @test size(ψ) == (3,)
    @test size(ψ, 1) == 3
    @test length(W) == 3
    @test size(W) == (3,)
    @test size(W, 1) == 3
end

@testset "QMpo CPU transfer" begin
    W = simple_dense_mpo(Float64, Site[1 // 2, 2])
    @test move_to_CPU!(W) === W
    @test !W.onGPU
    @test which_device(W) == Set((:CPU,))

    if CUDA.functional()
        @test move_to_CUDA!(W) === W
        @test W.onGPU
        @test which_device(W) == Set((:GPU,))
        @test move_to_CPU!(W) === W
        @test !W.onGPU
        @test which_device(W) == Set((:CPU,))
    end
end

@testset "Temporary zipper bond dimension does not overflow" begin
    @test SpinGlassTensors._saturating_mul(2, typemax(Int)) == typemax(Int)
    @test SpinGlassTensors._saturating_mul(typemax(Int), 3) == typemax(Int)
    @test SpinGlassTensors._saturating_mul(Int32(3), Int16(7)) == 21
    @test SpinGlassTensors._saturating_mul(3, 7) == 21
    @test_throws ArgumentError SpinGlassTensors._saturating_mul(-1, 2)
end

@testset "Zipper default temporary dimension" begin
    W = simple_dense_mpo(Float64, Site[1, 2])
    ψ = IdentityQMps(Float64, Dict(1 => 2, 2 => 2), 1; onGPU = false)
    canonise!(ψ, :left)

    # Dcut and Dtemp_multiplier deliberately retain their defaults. Before
    # saturating the multiplication, the temporary dimension overflowed to -2.
    out = zipper(W, ψ; iters_rand = 1, iters_svd = 0, iters_var = 0)

    @test length(out) == 2
    @test is_consistent(out)
    @test dot(out, out) ≈ 1.0
end
