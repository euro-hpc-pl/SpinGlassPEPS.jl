const example_solution_short = Solution(
    [-1.5, -1.2, -1.3],
    [[1, 2], [0, 1], [3, 4]],
    [0.25, 0.1, 0.2],
    [1, 1, 1],
    0.666,
    [Droplet[], Droplet[], Droplet[]],
    [[0], [0], [0]],
)

const example_degenerate_solution = Solution(
    [-1.5, -1.2, -1.5, -1.6, -1.1],
    [[1, 2, 3], [1, 3, 1], [3, 4, 1], [1, 4, 2], [4, 2, 1]],
    [0.23, 0.1, 0.23, 0.25, 0.05],
    [1, 1, 1, 1, 1],
    0.22,
    [Droplet[], Droplet[], Droplet[], Droplet[], Droplet[]],
    [[0], [0], [0], [0], [0]],
)

@testset "Bounding solution of size ≥ max_states gives solution of max_states length" begin
    bounded = bound_solution(example_solution_short, 2, 0.0)
    @test length(bounded.energies) == 2
    @test length(bounded.states) == 2
    @test length(bounded.probabilities) == 2
    @test length(bounded.degeneracy) == 2

end

@testset "Bounding solution of size < max_states gives the same solution (up to permutation)" begin
    bounded = bound_solution(example_solution_short, 10, 0.0)
    @test bounded.energies == [-1.5, -1.2, -1.3]
    @test bounded.probabilities == [0.25, 0.1, 0.2]
    @test bounded.states == [[1, 2], [0, 1], [3, 4]]
    @test bounded.degeneracy == [1, 1, 1]
end

@testset "Bounding solution clips at correct probability" begin
    @testset "when max_states=$(max_states)" for (max_states, expected_prob) ∈
                                                 [(1, 0.23), (2, 0.23), (3, 0.22)]
        bounded = bound_solution(example_degenerate_solution, max_states, 0.0)
        @test bounded.largest_discarded_probability == expected_prob
        @test all(bounded.probabilities .≥ expected_prob)
    end
end


@testset "Bounding solution preserves correspondence between energies, states and probabilities" begin
    sorted_idx = sortperm(example_degenerate_solution.probabilities, rev = true)
    for max_states ∈ 1:4 # problem for max_states = 5 
        bounded = bound_solution(example_degenerate_solution, max_states, 0.0)
        @test bounded.energies ==
              example_degenerate_solution.energies[sorted_idx][1:max_states]
        @test bounded.probabilities ==
              example_degenerate_solution.probabilities[sorted_idx][1:max_states]
        @test bounded.states == example_degenerate_solution.states[sorted_idx][1:max_states]
    end
end
