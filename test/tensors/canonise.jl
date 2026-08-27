D = 16

sites = [1, 3 // 2, 2, 5 // 2, 3, 7 // 2, 4]
d = [1, 2, 2, 2, 4, 2, 2]

id = Dict(j => d[i] for (i, j) in enumerate(sites))

@testset "Random QMps ($T)" for T in (Float32, Float64)
    checks = CUDA.functional() ? (true, false) : (false, ) 
    for toCUDA ∈ checks
        ψ = rand(QMps{T}, id, D)
        ϕ = rand(QMps{T}, id, D)
        @test is_consistent(ψ)
        @test is_consistent(ϕ)

        if toCUDA
            ψ = move_to_CUDA!(ψ)
            ϕ = move_to_CUDA!(ϕ)
            @test is_consistent(ψ)
            @test is_consistent(ϕ)
        end

        @testset "is left normalized" begin
            canonise!(ψ, :left)
            @test is_consistent(ψ)
            @test is_left_normalized(ψ)
            @test dot(ψ, ψ) ≈ one(T)
        end

        @testset "is right normalized" begin
            canonise!(ϕ, :right)
            @test is_consistent(ϕ)
            @test is_right_normalized(ϕ)
            @test dot(ϕ, ϕ) ≈ one(T)
        end
    end
end

@testset "Measure spectrum ($T)" for T in (Float32, Float64)
    svd_mps = TensorMap{T}(
        1 => T[
            -0.694389933025747 -0.7195989305943268;;;
            0.7195989305943268 -0.6943899330257469
        ],
        2 => T[0.7071067811865477; 0.0;;; -7.850461536237973e-17; 0.7071067811865477],
    )
    ψ = QMps(svd_mps)
    @test is_left_normalized(ψ)
    A = measure_spectrum(ψ)
    @test A[1] ≈ [1.0]
    @test A[2] ≈ [0.7071067811865476, 0.7071067811865475]
end
