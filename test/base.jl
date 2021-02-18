@testset "MPS" begin

D = 10
d = 4
sites = 5
T = ComplexF64

@testset "Random MPS with the same physical dimension" begin

    ψ = randn(MPS{T}, sites, D, d)

    @testset "has correct number of sites" begin
        @test length(ψ) == sites 
        @test size(ψ) == (sites, )      
    end
 
    @testset "has correct type" begin
        @test eltype(ψ) == T       
    end

    @testset "has correct rank" begin
        @test rank(ψ) == Tuple(fill(d, 1:sites))      
    end

    @testset "has correct bonds" begin
        @test bond_dimension(ψ) ≈ D     
        @test verify_bonds(ψ) === nothing
    end

    @testset "is equal to itself" begin
        @test ψ == ψ
        @test ψ ≈ ψ
    end

    @testset "is equal to its copy" begin
        ϕ = copy(ψ)
        @test ϕ == ψ
        @test ϕ ≈ ψ
    end
end 

@testset "Random MPS with varying physical dimension" begin

    dims = (3, 2, 5, 4)
    ψ = randn(MPS{T}, D, dims)
    
    @testset "has correct number of sites" begin
        n = length(dims)
        @test length(ψ) == n
        @test size(ψ) == (n, )      
    end
 
    @testset "has correct type" begin
        @test eltype(ψ) == T       
    end

    @testset "has correct rank" begin
        @test rank(ψ) == dims      
    end

    @testset "has correct bonds" begin
        @test bond_dimension(ψ) ≈ D     
        @test verify_bonds(ψ) === nothing
    end

    @testset "is equal to itself" begin
        @test ψ == ψ
        @test ψ ≈ ψ
    end

    @testset "is equal to its copy" begin
        ϕ = copy(ψ)
        @test ϕ == ψ
        @test ϕ ≈ ψ
    end
end

@testset "Random MPO with the same physical dimension" begin

    W = randn(MPO{T}, sites, D, d)

    @testset "has correct number of sites" begin
        @test length(W) == sites 
        @test size(W) == (sites, )      
    end
 
    @testset "has correct type" begin
        @test eltype(W) == T       
    end

    @testset "is equal to itself" begin
        @test W == W
        @test W ≈ W
    end

    @testset "is equal to its copy" begin
        U = copy(W)
        @test U == W
        @test U ≈ W
    end
end 

# @testset "Random MPO with varying physical dimension" begin

#     dims = (3, 2, 5, 4)
#     W = randn(MPO{T}, D, dims)
    
#     @testset "has correct number of sites" begin
#         n = length(dims)
#         @test length(W) == n
#         @test size(W) == (n, )      
#     end
 
#     @testset "has correct type" begin
#         @test eltype(W) == T       
#     end

#     @testset "is equal to itself" begin
#         @test W == W
#         @test W ≈ W
#     end

#     @testset "is equal to its copy" begin
#         U = copy(W)
#         @test U == W
#         @test U ≈ W
#     end
# end

@testset "MPS from tensor" begin
    ϵ = 1E-14

    dims = (2,3,4,3,5)
    sites = length(dims)
    A = randn(T, dims)

    ψ = MPS(A, :right)

    @test norm(ψ) ≈ 1
    @test_nowarn verify_bonds(ψ)
    @test_nowarn verify_physical_dims(ψ, dims)
    @test is_right_normalized(ψ)

    # from here - move to the attic
    AA = tensor(ψ)

    @test rank(ψ) == size(AA)
    @test norm(AA) ≈ 1
    @test size(AA) == size(A)

    vA = vec(A)
    nA = norm(vA)
    @test abs(1 - abs(dot(vec(AA), vA ./ nA))) < ϵ
    #@test AA ≈ A ./ norm(A) # this is true "module phase"

    B = randn(T, dims...)
    ϕ = MPS(B, :left)

    @test norm(ϕ) ≈ 1
    @test_nowarn verify_bonds(ϕ)
    @test_nowarn verify_physical_dims(ϕ, dims)
    @test is_left_normalized(ϕ)

    BB = tensor(ϕ)

    @test rank(ϕ) == size(BB)
    @test norm(BB) ≈ 1
    @test sqrt(sum(abs.(B) .^ 2)) ≈ norm(B)

    vB = vec(B)
    nB = norm(vB)
    @test abs(1 - abs(dot(vec(BB), vB ./ nB))) < ϵ
    #@test BB ≈ B ./ norm(B) # this is true "module phase"

    χ = MPS(A, :left)

    @test norm(χ) ≈ 1
    @test abs(1 - abs(dot(ψ, χ))) < ϵ
end

end
