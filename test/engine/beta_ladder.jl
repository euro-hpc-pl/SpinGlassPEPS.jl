# Inverse-temperature ladder with warm-started boundary MPS.
#
# Correctness bar: a warm-started ladder must return the same energies as a
# ladder of independent solves. Warm starting changes how each row's boundary MPS
# is *reached* (variational compression from the previous β instead of an exact
# product then truncation), not what it approximates, so a discrepancy means the
# variational step is converging somewhere it should not.

using SpinGlassPEPS
using SpinGlassPEPS.SpinGlassEngine: no_merge
using Test
using CUDA
using Logging

onGPU = CUDA.functional()

# Large enough that its boundary MPS actually need truncating; the 3x4x3
# pathological instance is contracted exactly and cannot exercise any of this.
const LADDER_INSTANCE =
    joinpath(@__DIR__, "instances", "chimera_droplets", "128power", "001.txt")
const LM, LN, LT = 4, 4, 8

ladder_hamiltonian() = potts_hamiltonian(
    ising_graph(LADDER_INSTANCE),
    2^6;
    spectrum = brute_force,
    cluster_assignment_rule = super_square_lattice((LM, LN, LT)),
)

function ladder_contractor(potts_h, bond_dim; strategy = SVDTruncate, beta = 1.0)
    MpsContractor(
        strategy,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(
            LM,
            LN,
            potts_h,
            all_lattice_transformations[1],
        ),
        MpsParameters{Float64}(; bond_dim = bond_dim, num_sweeps = 4);
        onGPU = onGPU,
        beta = beta,
        graduate_truncation = true,
    )
end

@testset "set_beta! retargets and evicts" begin
    potts_h = ladder_hamiltonian()
    ctr = ladder_contractor(potts_h, 16)
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)

    low_energy_spectrum(ctr, sparams, no_merge; show_progress = false, retain_mps = true)
    # Rows are snapshotted during preprocessing, since the search consumes each
    # row's MPS as it absorbs it.
    @test !isempty(ctr.guess)
    retained = sort(collect(keys(ctr.guess)))
    @test retained == collect(2:ctr.peps.nrows+1)
    # Guesses must be on the host: keeping one MPS per row resident would defeat
    # the point of accounting for device memory.
    @test all(ψ -> !ψ.onGPU, values(ctr.guess))

    set_beta!(ctr, 2.0)
    @test ctr.beta == 2.0
    @test isempty(ctr.cache.mps)
    @test isempty(ctr.cache.mpo)
    @test sort(collect(keys(ctr.guess))) == retained   # warm start keeps them

    set_beta!(ctr, 3.0; warm_start = false)
    @test ctr.beta == 3.0
    @test isempty(ctr.guess)                            # ... and this drops them

    # Without retain_mps nothing is kept, so a later step cannot warm start.
    ctr2 = ladder_contractor(potts_h, 16)
    low_energy_spectrum(ctr2, sparams, no_merge; show_progress = false)
    @test isempty(ctr2.guess)
end

@testset "warm-started ladder matches independent solves" begin
    potts_h = ladder_hamiltonian()
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    schedule = [0.5, 1.0, 2.0, 3.0]

    cold = beta_ladder(ladder_contractor(potts_h, 16), schedule, sparams; warm_start = false)
    warm = beta_ladder(ladder_contractor(potts_h, 16), schedule, sparams; warm_start = true)

    @test [s.beta for s ∈ cold.steps] == schedule
    @test [s.beta for s ∈ warm.steps] == schedule
    @test all(s -> s.error === nothing, cold.steps)
    @test all(s -> s.error === nothing, warm.steps)

    # Warm starting must actually have engaged, or this test proves nothing.
    @test [s.warm_started for s ∈ warm.steps] == [false, true, true, true]
    @test all(s -> !s.warm_started, cold.steps)

    # The whole point: same answer, reached more cheaply.
    @test [s.energy for s ∈ warm.steps] ≈ [s.energy for s ∈ cold.steps]

    # Every rung solved this instance, so selection picks a real rung and the
    # solution it points at is the one whose energy was reported.
    @test warm.selected_index != 0
    sol = selected_solution(warm)
    @test first(sol.energies) ≈ warm.steps[warm.selected_index].energy
    @test length(warm.solutions) == length(schedule)
end

@testset "ladder error guard and selection" begin
    potts_h = ladder_hamiltonian()
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    schedule = [1.0, 2.0, 3.0]

    # A guard of zero rejects any rung that discarded anything at all. Selection
    # must fall back to the untrusted rungs rather than reporting nothing.
    # `warm_start = false`: the guard is only meaningful for cold builds, since a
    # warm start never performs a truncating factorization and so reports no
    # discarded weight whatever its accuracy.
    strict = beta_ladder(
        ladder_contractor(potts_h, 4),
        schedule,
        sparams;
        max_discarded = 0.0,
        warm_start = false,
    )
    @test !all(s -> s.trusted, strict.steps)
    @test strict.selected_index != 0
    @test selected_solution(strict) !== nothing

    # A permissive guard trusts everything and selects the lowest energy. Infinite
    # guard means no guard, so warm starting is not misleading here and no warning
    # is expected.
    loose = beta_ladder(
        ladder_contractor(potts_h, 16),
        schedule,
        sparams;
        max_discarded = Inf,
    )
    @test all(s -> s.trusted, loose.steps)
    energies = [s.energy for s ∈ loose.steps]
    @test loose.steps[loose.selected_index].energy ≈ minimum(energies)

    # Early exit: with a guard of zero and stop_when_untrusted, the ladder must
    # not climb past the first untrusted rung. Asserted on the returned steps
    # rather than on the info message, which the suite's `disable_logging` hides.
    stopped = beta_ladder(
        ladder_contractor(potts_h, 4),
        schedule,
        sparams;
        max_discarded = 0.0,
        stop_when_untrusted = true,
        warm_start = false,
    )
    @test length(stopped.steps) < length(schedule)
    @test !last(stopped.steps).trusted
    # Rungs beyond the stopping point were never attempted, so they hold no
    # solution.
    @test all(isnothing, stopped.solutions[length(stopped.steps)+1:end])
end

@testset "error guard warns when combined with warm start" begin
    # The guard cannot see a warm-started rung's error, so pairing them must warn
    # rather than silently rate every warmed rung as trustworthy.
    potts_h = ladder_hamiltonian()
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    @test_logs (:warn,) match_mode = :any beta_ladder(
        ladder_contractor(potts_h, 8),
        [1.0, 2.0],
        sparams;
        max_discarded = 1e-3,
        warm_start = true,
    )
    # Cold, or with no finite guard, is unambiguous and must stay quiet.
    @test_logs min_level = Logging.Warn beta_ladder(
        ladder_contractor(potts_h, 8),
        [1.0, 2.0],
        sparams;
        max_discarded = 1e-3,
        warm_start = false,
    )
end

@testset "ladder argument validation" begin
    potts_h = ladder_hamiltonian()
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    ctr = ladder_contractor(potts_h, 8)

    @test_throws ArgumentError beta_ladder(ctr, Float64[], sparams)
    @test_throws ArgumentError beta_ladder(ctr, [1.0, 0.0], sparams)
    @test_throws ArgumentError beta_ladder(ctr, [-1.0], sparams)

    # A decreasing schedule is honoured but warned about: warm starting assumes
    # consecutive betas are close and the guard reads as "how far up we got".
    @test_logs (:warn,) match_mode = :any beta_ladder(ctr, [2.0, 1.0], sparams)
end

@testset "warm start on a geometry with non-uniform local dimensions" begin
    # The 128power instance has uniform cluster dimensions, so an MPO row's :up
    # and :down local dimensions coincide there and a guess-compatibility check
    # against the wrong side would pass by accident. This instance's do not
    # (row 2: :up = [2,4,1,1], :down = [1,2,1,1]), so it pins down that the
    # bottom boundary MPS is dimensioned by :up. A regression would silently stop
    # warm-starting rather than give wrong answers, hence the explicit assertion
    # that it engaged.
    pm, pn, pt = 3, 4, 3
    potts_h = potts_hamiltonian(
        ising_graph(joinpath(@__DIR__, "instances", "pathological", "chim_3_4_3.txt"));
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((pm, pn, pt)),
    )
    mk() = MpsContractor(
        SVDTruncate,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(
            pm,
            pn,
            potts_h,
            all_lattice_transformations[1],
        ),
        MpsParameters{Float64}(; bond_dim = 8, num_sweeps = 1);
        onGPU = onGPU,
        beta = 1.0,
        graduate_truncation = true,
    )
    sparams = SearchParameters(; max_states = 2^6, cutoff_prob = 1e-4)
    schedule = [1.0, 2.0]

    cold = beta_ladder(mk(), schedule, sparams; warm_start = false)
    warm = beta_ladder(mk(), schedule, sparams; warm_start = true)
    @test [s.warm_started for s ∈ warm.steps] == [false, true]
    @test [s.energy for s ∈ warm.steps] ≈ [s.energy for s ∈ cold.steps]
end

@testset "warm start works under Zipper" begin
    # Zipper reaches each row through the randomized zipper rather than an exact
    # product, so it takes a different warm-start branch and needs its own check.
    potts_h = ladder_hamiltonian()
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    schedule = [1.0, 2.0]

    cold = beta_ladder(
        ladder_contractor(potts_h, 16; strategy = Zipper),
        schedule,
        sparams;
        warm_start = false,
    )
    warm = beta_ladder(
        ladder_contractor(potts_h, 16; strategy = Zipper),
        schedule,
        sparams;
        warm_start = true,
    )
    @test [s.warm_started for s ∈ warm.steps] == [false, true]
    @test [s.energy for s ∈ warm.steps] ≈ [s.energy for s ∈ cold.steps]
end
