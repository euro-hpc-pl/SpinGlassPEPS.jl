TEST_CASES = [
    # projectors, expected_fused, expected_transitions
    (([1 1; 0 1], [1 0; 1 1]), [1 0; 0 1], [[1 1; 0 1], [1 0; 1 1]]),
    (([1 1; 0 0], [1 0; 0 1]), [1 0; 0 1], [[1 1; 0 0], [1 0; 0 1]]),
    (
        ([1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]),
        [1 0 0; 0 1 0; 0 0 1],
        [[1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]],
    ),
    (
        ([1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]),
        [1 0 0; 0 1 0; 0 0 1],
        [[1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]],
    ),
]


@testset "fuse_projectors correctly fuses projectors." for (
    projectors,
    expected_fused,
    expected_transactions,
) ∈ TEST_CASES
    fused, transitions = fuse_projectors(projectors)
    @test fused == expected_fused
    @test transitions == expected_transactions
end

@testset "normalize_probability properly deals with negative values" begin
    values = [1.0, -1.0, 15.324, -2.2134, 0.123, 0.0]
    new_values = [2.2134, 2.2134, 15.324, 2.2134, 2.2134, 2.2134]
    #  sum = 26.391
    expected_prob = [
        0.08386950096623849,
        0.08386950096623849,
        0.5806524951688076,
        0.08386950096623849,
        0.08386950096623849,
        0.08386950096623849,
    ]

    @test normalize_probability(values) ≈ expected_prob ≈ new_values / sum(new_values)
end
