@testset "state indexing" begin
    σ = (1, -1, -1, 1, 1)
    d = 2^length(σ)
    x = rand(2, 2, d)
    @test (@state x[1, 1, σ]) == x[1, 1, 20]
end

@testset "Reshape by rows" begin
    A = [1 2; 3 4]
    @test A[1,2] == 2
    display(A)
    B = reshape_row(A, (2,2))
    display(B)    
    @test B[1,1] == 1
    @test B[1,2] == 3
    @test B[2,1] == 2
    @test B[2,2] == 4

    C = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    display(C)
    D = reshape_row(C, (2,2,2,2))
    display(D)
    @test D[:, :, 1, 1] == [1 2 ; 3 4]
    @test D[:, :, 1, 2] == [5 6 ; 7 8]
    @test D[:, :, 2, 1] == [9 10 ; 11 12]
    @test D[:, :, 2, 2] == [13 14 ; 15 16]
end