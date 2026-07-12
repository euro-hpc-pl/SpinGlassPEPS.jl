using Test
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors


@testset "count_ones() method" begin
    # Julia program to illustrate the use of count_ones() method
    # Getting number of ones present in the binary representation of the specified number.
    for t in [0, 1, 15, 20]
        @test count_ones(t) == count(==('1'), bitstring(t))
    end
end

@testset "XOR Operation Tests" begin
    @test 0 ⊻ 0 == xor(0, 0) == 0
    @test 0 ⊻ 1 == xor(0, 1) == 1
    @test 1 ⊻ 0 == xor(1, 0) == 1
    @test 1 ⊻ 1 == xor(1, 1) == 0
    @test 10 ⊻ 5 == xor(10, 5) == 15
    @test 15 ⊻ 5 == xor(15, 5) == 10
    @test 255 ⊻ 85 == xor(255, 85) == 170
    @test 170 ⊻ 85 == xor(170, 85) == 255

    binary_string = "1111"
    integer_value = parse(Int, binary_string, base = 2)
    @test 10 ⊻ 5 == xor(10, 5) == 15
end

flip1 = Flip([1, 3, 4, 5], [2, 3, 5, 6], [1, 1, 0, 1], [1, 1, 0, 1])
flip2 = Flip([1, 3, 4, 5], [1, 2, 3, 4], [1, 1, 0, 1], [1, 1, 0, 1])
flip3 = Flip([1, 3, 4, 5], [2, 3, 5, 6], [1, 1, 0, 0], [1, 1, 0, 0])

@testset "Hamming Distance Calculation" begin
    @test hamming_distance(flip1, :Ising) == 3
    @test hamming_distance(flip2, :Ising) == 3
    @test hamming_distance(flip3, :Ising) == 2
    @test hamming_distance(flip1, :RMF) == 3
    @test hamming_distance(flip2, :RMF) == 3
    @test hamming_distance(flip3, :RMF) == 2
end

@testset "Hamming distance between two droplets" begin
    @test hamming_distance(flip1, flip2, :Ising) == 0
    @test hamming_distance(flip2, flip3, :Ising) == 1
    @test hamming_distance(flip1, flip2, :RMF) == 3
    @test hamming_distance(flip2, flip3, :RMF) == 3

end
