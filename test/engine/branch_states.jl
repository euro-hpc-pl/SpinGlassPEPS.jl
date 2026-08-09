# `branch_states` expands each surviving partial configuration by every state of
# the next cluster's local basis.
#
# Its ordering is load-bearing: `branch_solution` pairs the returned vector
# index-for-index with `branch_energies` and `branch_probability`, so a permutation
# would silently attach the wrong energy to a state rather than fail loudly. These
# tests pin the contract independently of the implementation, which was rewritten
# to stop materializing the expanded set twice.

using SpinGlassPEPS
using SpinGlassPEPS.SpinGlassEngine: branch_states
using Test
using Random

# Reference implementation: the expansion stated declaratively. The local basis
# must vary fastest within each parent configuration.
reference_branch_states(basis, parents) =
    [vcat(p, b) for p ∈ parents for b ∈ basis]

@testset "branch_states expansion and ordering" begin
    # Explicit small case, so the expected order is visible rather than derived.
    @test branch_states([7, 8], [[1, 2], [3, 4]]) ==
          [[1, 2, 7], [1, 2, 8], [3, 4, 7], [3, 4, 8]]

    # First node of a search: one parent, which is the empty configuration.
    @test branch_states([1, 2, 3], [Int[]]) == [[1], [2], [3]]

    # Single-element basis leaves the parent count unchanged.
    @test branch_states([5], [[1], [2], [3]]) == [[1, 5], [2, 5], [3, 5]]

    # Result length is |basis| * |parents|, and every entry is one longer than its
    # parent.
    for (nb, np, lp) ∈ ((1, 1, 0), (3, 1, 2), (1, 4, 3), (5, 6, 4))
        basis = collect(1:nb)
        parents = [collect(1:lp) for _ = 1:np]
        got = branch_states(basis, parents)
        @test length(got) == nb * np
        @test all(length.(got) .== lp + 1)
    end

    # Agreement with the declarative reference over random shapes.
    Random.seed!(0xC0FFEE)
    for _ = 1:200
        basis = rand(1:9, rand(1:6))
        lstate = rand(0:5)
        parents = [rand(1:9, lstate) for _ = 1:rand(1:7)]
        @test branch_states(basis, parents) == reference_branch_states(basis, parents)
    end

    # Parents must not be aliased into the result: mutating a returned state must
    # not disturb its siblings or the input.
    parents = [[1, 2], [3, 4]]
    got = branch_states([9], parents)
    got[1][1] = 99
    @test parents == [[1, 2], [3, 4]]
    @test got[2] == [3, 4, 9]

    # Degenerate input: no parents means nothing to expand.
    @test branch_states([1, 2], Vector{Int}[]) == Vector{Int}[]
end
