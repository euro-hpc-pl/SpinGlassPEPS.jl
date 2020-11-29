@testset "MPS" begin

D = 10
d = 4
sites = 5
T = ComplexF64

@testset "Random MPS" begin
    ψ = randn(MPS{T}, sites, D, d)
    @test verify_bonds(ψ) == nothing

    @test ψ == ψ
    @test ψ ≈ ψ

    @test length(ψ) == sites
    @test size(ψ) == (sites, )
    @test eltype(ψ) == ComplexF64
    @test rank(ψ) == Tuple(fill(d, 1:sites))
    @test bond_dimension(ψ) ≈ D

    ϕ = copy(ψ) 
    @test ϕ == ψ
    @test ϕ ≈ ψ

    show(ψ)

    dims = (3, 2, 5, 4) 
    @info "Veryfing ψ of arbitrary rank" dims

    ψ = randn(MPS{T}, D, dims)
    @test verify_bonds(ψ) == nothing

    @test ψ == ψ
    @test ψ ≈ ψ

    @test length(ψ) == length(dims)
    @test size(ψ) == (length(dims), )
    @test eltype(ψ) == ComplexF64
    @test rank(ψ) == dims
    @test bond_dimension(ψ) ≈ D

    ϕ = copy(ψ) 
    @test ϕ == ψ
    @test ϕ ≈ ψ

    show(ψ)
end

@testset "Random MPO" begin
    O = randn(MPO{T}, sites, D, d)

    @test O == O
    @test O ≈ O

    @test length(O) == sites
    @test size(O) == (sites, )
    @test eltype(O) == ComplexF64

    P = copy(O)
    @test P == O 
    @test P ≈ O 
end

@testset "Reshaping (row-wise)" begin
    vec = Vector(1:6)

    A = reshape_row(vec, (2, 3))
    B = [1 2 3; 4 5 6]

    @test A ≈ B
end 

@testset "Basic vector to tensor reshaping" begin
    dims = (2, 3, 4, 5)
    states = [randn(T, d) for d ∈ dims] 
    vec = kron(states...)

    ψ = tensor(MPS(states))
    ϕ = reshape_row(vec, dims)

    @test ψ ≈ ϕ
end 

@testset "MPS from tensor" begin
    ϵ = 1E-14

    dims = (2,3,4,3,5) 
    sites = length(dims)
    A = randn(T, dims) 

    @test sqrt(sum(abs.(A) .^ 2)) ≈ norm(A)

    @test ndims(A) == sites 
    @test size(A) == dims

    ψ = MPS(A, :right)

    @test norm(ψ) ≈ 1
    @test_nowarn verify_bonds(ψ)
    @test_nowarn verify_physical_dims(ψ, dims)
    @test is_right_normalized(ψ)
    show(ψ)

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
    show(ϕ)

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