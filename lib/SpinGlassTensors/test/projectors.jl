
@testset "Add and get from pool of projectors" begin
    @testset "Start with empty pool and add elements to it" begin
        lp = PoolOfProjectors{Int64}()
        @test length(lp) == 0

        p1 = [1, 1, 2, 2, 3, 3]
        p2 = [1, 2, 1, 3]
        k = add_projector!(lp, p1)
        @test k == 1
        @test length(lp) == 1

        k = add_projector!(lp, p1)
        @test k == 1
        @test length(lp) == 1

        k = add_projector!(lp, p2)
        @test k == 2
        @test length(lp) == 2

        @test get_projector!(lp, 1) == p1
        @test get_projector!(lp, 2) == p2

        @test length(lp, 1) == 6
        @test length(lp, 2) == 4
        @test size(lp, 1) == 3
        @test size(lp, 2) == 3

        empty!(lp, lp.default_device)
        @test length(lp) == 0
    end

    @testset "Different devices" begin
        checks = CUDA.functional() ? (true, false) : (false) 
        for T ∈ [Int16, Int32, Int64]
            for toCUDA ∈ checks
                lp = PoolOfProjectors{T}()
                p1 = Vector{T}([1, 1, 2, 2, 3, 3])
                p2 = Vector{T}([1, 2, 1, 3])
                k = add_projector!(lp, p1)
                k = add_projector!(lp, p2)

                if toCUDA
                    @test typeof(get_projector!(lp, 1, :GPU)) <: CuArray{T,1}
                    @test length(lp, :GPU) == 1

                    @test typeof(get_projector!(lp, 1, :GPU)) <: CuArray{T,1}
                    @test length(lp, :GPU) == 1
    
                    @test typeof(get_projector!(lp, 2, :GPU)) <: CuArray{T,1}
                    @test length(lp, :GPU) == 2
                end

                @test typeof(get_projector!(lp, 2, :CPU)) <: Array{T,1}
                @test length(lp, :CPU) == 2   
            end    
        end
    end
end
