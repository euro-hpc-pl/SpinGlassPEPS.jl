# Concurrent transformation sweep and its device-memory governor.
#
# Correctness bar: a sweep must return exactly what the serial published loop
# returns. Everything else here (budget arithmetic, seeding, diagnostics) is
# machinery in service of that.

using SpinGlassPEPS
using SpinGlassPEPS.SpinGlassEngine: reserve!, release!, no_merge
using SpinGlassPEPS.SpinGlassTensors: canonise_truncate!
using Base.ScopedValues: with
using Test
using CUDA
using Random

onGPU = CUDA.functional()

instance = joinpath(@__DIR__, "instances", "pathological", "chim_3_4_3.txt")
m, n, t = 3, 4, 3
Dcut, β, max_states = 8, 1.0, 2^6

function build_hamiltonian()
    potts_hamiltonian(
        ising_graph(instance),
        spectrum = full_spectrum,
        cluster_assignment_rule = super_square_lattice((m, n, t)),
    )
end

# The serial protocol from the published examples, for reference.
function serial_sweep(potts_h, params, sparams; transforms = all_lattice_transformations)
    map(collect(transforms)) do transform
        net = PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(
            m,
            n,
            potts_h,
            transform,
        )
        ctr = MpsContractor(
            SVDTruncate,
            net,
            params;
            onGPU = onGPU,
            beta = β,
            graduate_truncation = true,
        )
        sol, _ = low_energy_spectrum(ctr, sparams, no_merge; show_progress = false)
        first(sol.energies)
    end
end

@testset "DeviceBudget admission control" begin
    @test_throws ArgumentError DeviceBudget(0)
    @test_throws ArgumentError DeviceBudget(-1)

    b = DeviceBudget(1000)
    @test reserve!(b, 400) == 400
    @test reserve!(b, 400) == 400
    @test b.reserved == 800
    @test b.peak_reserved == 800
    @test b.admissions == 2
    @test b.waits == 0

    # A third 400 does not fit; it must block until something is released.
    blocked = Threads.@spawn reserve!(b, 400)
    # Give the task a chance to reach the wait. Without extra threads it will
    # only get there once we yield, which `sleep` does.
    sleep(0.2)
    @test b.reserved == 800          # still blocked
    release!(b, 400)
    wait(blocked)
    @test b.reserved == 800
    @test b.waits == 1
    release!(b, 400)
    release!(b, 400)
    @test b.reserved == 0

    # Oversized requests must not deadlock: they run alone.
    b2 = DeviceBudget(100)
    @test reserve!(b2, 10_000) == 10_000
    release!(b2, 10_000)
    @test b2.reserved == 0

    # Zero-sized reservations are free and are not counted.
    b3 = DeviceBudget(100)
    @test reserve!(b3, 0) == 0
    @test b3.admissions == 0
    release!(b3, 0)
end

@testset "sweep reproduces the serial transformation loop" begin
    potts_h = build_hamiltonian()
    params = MpsParameters{Float64}(; bond_dim = Dcut, num_sweeps = 1)
    sparams = SearchParameters(; max_states = max_states, cutoff_prob = 1e-4)

    expected = serial_sweep(potts_h, params, sparams)

    build = transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )

    # Explicit concurrency, not the default: `:auto` is 1 on a GPU (fanning these
    # solves over one device measures slower than sequencing them), so relying on
    # the default here would quietly stop testing the concurrent path at all.
    sweep = sweep_transformations(
        build,
        sparams;
        merge_strategy = _ -> no_merge,
        concurrency = 4,
    )

    @test length(sweep.solutions) == length(all_lattice_transformations)
    @test sweep.report.failures == 0
    @test all(s -> s !== nothing, sweep.solutions)

    got = [first(s.energies) for s ∈ sweep.solutions]
    @test got ≈ expected

    # The best solution must be the minimum over transformations, and must be a
    # genuine Solution the caller can use.
    @test sweep.best_index != 0
    best = best_solution(sweep)
    @test first(best.energies) ≈ minimum(expected)
    # States are indexed by cluster, not by spin: this lattice has m*n clusters.
    @test length(first(best.states)) == m * n

    # This instance is solved exactly by every transformation, so the sweep
    # should report full consensus and no spread. That is the signal the
    # diagnostics exist to provide.
    @test sweep.report.energy_spread ≈ 0 atol = 1e-8
    @test sweep.report.consensus == length(all_lattice_transformations)

    # The serial path (what `:auto` selects on a GPU) must agree with both.
    seq = sweep_transformations(build, sparams; concurrency = 1)
    @test [first(s.energies) for s ∈ seq.solutions] ≈ expected
    @test seq.report.max_concurrency == 1
end

@testset "sweep is reproducible and seed-controlled" begin
    potts_h = build_hamiltonian()
    params = MpsParameters{Float64}(; bond_dim = Dcut, num_sweeps = 1)
    sparams = SearchParameters(; max_states = max_states, cutoff_prob = 1e-4)

    # Zipper draws a random sketch, so this is the strategy that would go
    # non-deterministic under concurrency without per-task seeding.
    build = transform -> MpsContractor(
        Zipper,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )

    a = sweep_transformations(build, sparams; seed = 99)
    b = sweep_transformations(build, sparams; seed = 99)
    ea = [first(s.energies) for s ∈ a.solutions]
    eb = [first(s.energies) for s ∈ b.solutions]
    @test ea == eb
end

@testset "boundary-MPS truncation reaches the log" begin
    # The factorization-level contract is covered in the tensors group
    # (test/tensors/scoping.jl); what matters here is that a truncating
    # boundary-MPS sweep, which is several layers above `svd_fact`, reaches it
    # without any signature threading.
    log = TruncationLog()
    with(TRUNCATION_LOG => log) do
        canonise_truncate!(rand(QMps{Float64}, Dict(i => 8 for i = 1:6), 16), :left, 3, 0.0)
    end
    s = truncation_stats(log)
    @test s.count > 0
    @test s.discarded_sum > 0
    @test s.dims_kept < s.dims_offered
end

@testset "sweep reports contraction error" begin
    # A larger instance than the equivalence tests above: the 3x4x3 pathological
    # boundary MPS is exact (bond dims 1-2), so nothing is ever truncated there
    # and it cannot exercise this diagnostic.
    big = joinpath(@__DIR__, "instances", "chimera_droplets", "128power", "001.txt")
    bm, bn, bt = 4, 4, 8
    potts_h = potts_hamiltonian(
        ising_graph(big),
        2^6;
        spectrum = brute_force,
        cluster_assignment_rule = super_square_lattice((bm, bn, bt)),
    )
    sparams = SearchParameters(; max_states = 2^4, cutoff_prob = 1e-4)
    single = [all_lattice_transformations[1]]

    build(bond_dim) =
        transform -> MpsContractor(
            SVDTruncate,
            PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(
                bm,
                bn,
                potts_h,
                transform,
            ),
            MpsParameters{Float64}(; bond_dim = bond_dim, num_sweeps = 1);
            onGPU = onGPU,
            beta = 1.0,
            graduate_truncation = true,
        )

    tight = sweep_transformations(build(4), sparams; transformations = single)
    loose = sweep_transformations(build(32), sparams; transformations = single)

    tstat = tight.report.per_transform[1].truncation
    lstat = loose.report.per_transform[1].truncation

    @test tstat.count > 0
    @test 0 <= tstat.discarded_max <= 1
    @test tstat.dims_kept <= tstat.dims_offered

    # The point of the diagnostic: squeezing the bond dimension must show up as
    # more discarded weight and as bond-limited (rather than tolerance-limited)
    # truncations.
    @test tstat.discarded_sum > lstat.discarded_sum
    @test tstat.saturated > 0
    @test lstat.discarded_sum < 1e-8   # D=32 contracts this instance essentially exactly
    # ... and its bond-capped cuts drop only negligible weight, so none counts
    # as bond-limited: no change from raising D is expected at reported precision.
    @test lstat.saturated == 0

    # This instance is large enough to move the driver's free-memory figure, so
    # calibration must produce a real measurement and size the reservation from
    # it. (The tiny instance used elsewhere in this file cannot.)
    if onGPU
        @test loose.report.calibrated_peak > 0
        @test loose.report.reservation >= loose.report.calibrated_peak
        @test loose.report.capacity > loose.report.reservation
    end

    # With diagnostics off, nothing is recorded.
    off = sweep_transformations(
        build(4),
        sparams;
        transformations = single,
        diagnostics = false,
    )
    @test off.report.per_transform[1].truncation.count == 0
end

@testset "reservation policies" begin
    potts_h = build_hamiltonian()
    params = MpsParameters{Float64}(; bond_dim = Dcut, num_sweeps = 1)
    sparams = SearchParameters(; max_states = max_states, cutoff_prob = 1e-4)
    build = transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )
    pair = all_lattice_transformations[1:2]
    reference = serial_sweep(potts_h, params, sparams; transforms = pair)

    # Governor disabled entirely.
    none = sweep_transformations(
        build,
        sparams;
        transformations = pair,
        reservation = :none,
    )
    @test [first(s.energies) for s ∈ none.solutions] ≈ reference
    @test none.report.reservation == 0
    @test none.report.calibrated_peak == 0

    # Explicit reservation: no calibration run, so all transformations are
    # subject to the budget and none is marked as the calibration solve.
    fixed = sweep_transformations(
        build,
        sparams;
        transformations = pair,
        reservation = 64 * 2^20,
    )
    @test [first(s.energies) for s ∈ fixed.solutions] ≈ reference
    @test fixed.report.reservation == 64 * 2^20
    @test !any(r -> r.calibration, fixed.report.per_transform)
    @test fixed.report.max_concurrency >= 1

    # An absurd reservation must still complete — the oversized-request escape
    # hatch degrades to serial rather than deadlocking.
    huge = sweep_transformations(
        build,
        sparams;
        transformations = pair,
        reservation = 1 << 50,
    )
    @test [first(s.energies) for s ∈ huge.solutions] ≈ reference
    if onGPU
        # A reservation larger than the device forces one solve at a time.
        @test huge.report.max_concurrency == 1
    else
        # With no device there is no device memory to ration, so a nominal
        # reservation of any size must not constrain concurrency.
        @test huge.report.capacity > (1 << 50)
        @test huge.report.max_concurrency >= 1
    end

    # Calibration is the default: exactly one solve runs solo and is marked as
    # such. This instance is far too small to register against the driver's
    # free-memory granularity, so the governor is expected to stand down —
    # `sweep reports contraction error` covers the measurable case.
    cal = sweep_transformations(build, sparams; transformations = pair)
    @test [first(s.energies) for s ∈ cal.solutions] ≈ reference
    @test count(r -> r.calibration, cal.report.per_transform) == 1
    @test cal.report.calibration_time > 0
    @test cal.report.reservation >= 0
    @test cal.report.reservation >=
          Int(ceil(cal.report.calibrated_peak * 1.3))  # headroom honoured
end

@testset "argument validation and failure isolation" begin
    potts_h = build_hamiltonian()
    params = MpsParameters{Float64}(; bond_dim = Dcut, num_sweeps = 1)
    sparams = SearchParameters(; max_states = max_states, cutoff_prob = 1e-4)
    build = transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{SquareSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params;
        onGPU = onGPU,
        beta = β,
        graduate_truncation = true,
    )

    @test_throws ArgumentError sweep_transformations(
        build,
        sparams;
        transformations = LatticeTransformation[],
    )
    @test_throws ArgumentError sweep_transformations(
        build,
        sparams;
        transformations = all_lattice_transformations[1:1],
        vram_fraction = 1.5,
    )
    @test_throws ArgumentError sweep_transformations(
        build,
        sparams;
        transformations = all_lattice_transformations[1:1],
        reservation_headroom = 0.5,
    )
    @test_throws ArgumentError sweep_transformations(
        build,
        sparams;
        transformations = all_lattice_transformations[1:1],
        concurrency = :all,
    )
    @test_throws ArgumentError sweep_transformations(
        build,
        sparams;
        transformations = all_lattice_transformations[1:1],
        reservation = :measure,
    )

    # One transformation throwing must not lose the others: the sweep records
    # the failure and still returns the good solutions.
    calls = Threads.Atomic{Int}(0)
    flaky = function (transform)
        i = Threads.atomic_add!(calls, 1) + 1
        i == 2 && error("synthetic failure")
        build(transform)
    end
    pair = all_lattice_transformations[1:2]
    result = @test_logs (:error,) match_mode = :any sweep_transformations(
        flaky,
        sparams;
        transformations = pair,
        reservation = :none,
    )
    @test result.report.failures == 1
    @test count(s -> s !== nothing, result.solutions) == 1
    @test result.best_index != 0
    @test best_solution(result) !== nothing
end
