# sweep.jl: concurrent branch-and-bound over the lattice transformations, with
# an explicit device-memory governor.
#
# The published protocol solves the same instance once per element of
# `all_lattice_transformations` and keeps the best result: the search sweeps
# variables in a fixed order tied to the contraction order, so rotating and
# reflecting the lattice starts the sweep from different corners and materially
# stabilizes the outcome. Those eight solves are independent, and since the
# contraction caches became contractor-owned (no process-global memoization)
# nothing prevents running them at once. Until now the loop lived in user code
# and ran serially, which is the main reason the solver's time-to-solution
# compares poorly with samplers that parallelize trivially.
#
# The binding constraint is not CPU cores, it is VRAM. A single bond-32 solve on
# a Pegasus-scale instance can hold a few GiB of device memory, so a fixed
# eight-way fan-out reliably exhausts a consumer card. Concurrency is therefore
# gated by a *byte* budget rather than a task count:
#
#   1. one transformation is solved alone, under a provisional budget equal to
#      the slice it would get in the parallel phase, while its peak device usage
#      is measured. That run is not overhead: its solution counts, and it absorbs
#      the JIT compilation that all eight solves would otherwise duplicate under
#      contention;
#   2. the measured peak (times a headroom factor) becomes the per-solve
#      reservation, and the number of reservations that fit in the usable device
#      memory becomes the admission limit;
#   3. each admitted solve runs with `DEVICE_MEMORY_BUDGET` set to its
#      reservation, so `kernel_batch_size` sizes kernel intermediates against
#      that slice instead of against the shared free pool. Without this step the
#      reservation would be advisory only — every concurrent solve would measure
#      the same free memory and batch as if it owned all of it.

export DeviceBudget,
    SweepSolution,
    SweepReport,
    TransformReport,
    sweep_transformations,
    best_solution,
    reserve!,
    release!

using Base.ScopedValues: with
import Random

# ---------------------------------------------------------------------------
# Device-memory budget: a counting semaphore denominated in bytes.
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

A counting semaphore denominated in bytes, used to admit concurrent solves only
while their combined device-memory reservations fit in `capacity`.

A byte budget rather than a worker count, because the number of solves that fit
on a device is a property of the instance (bond dimension, cluster size,
geometry, element type), not of the machine's core count. On a 16 GiB card a
bond-16 Chimera sweep admits all eight transformations at once while a bond-32
Pegasus sweep admits two.

An oversized request (one larger than `capacity`) is admitted alone rather than
deadlocking, so a budget that turns out to be too small degrades to serial
execution instead of hanging.

# Fields
- `capacity::Int`: total bytes available to hand out.
- `reserved::Int`: bytes currently reserved.
- `peak_reserved::Int`: high-water mark of `reserved`, for reporting.
- `admissions::Int`: how many reservations were granted.
- `waits::Int`: how many reservations had to block first — the signal that the
  sweep was VRAM-bound rather than compute-bound.
"""
mutable struct DeviceBudget
    capacity::Int
    reserved::Int
    peak_reserved::Int
    admissions::Int
    waits::Int
    cond::Threads.Condition
end

function DeviceBudget(capacity::Integer)
    capacity > 0 || throw(ArgumentError("budget capacity must be positive, got $capacity"))
    DeviceBudget(Int(capacity), 0, 0, 0, 0, Threads.Condition(ReentrantLock()))
end

"""
$(TYPEDSIGNATURES)

Reserve `n` bytes, blocking until they fit alongside the reservations already
outstanding. Returns `n` so that the caller can pass the result to
[`release!`](@ref) unchanged.
"""
function reserve!(b::DeviceBudget, n::Integer)
    n = Int(n)
    n <= 0 && return 0
    lock(b.cond)
    try
        # `b.reserved == 0` is the oversized-request escape hatch: a solve that
        # cannot fit the budget at all still runs, just never alongside another.
        if !(b.reserved + n <= b.capacity || b.reserved == 0)
            b.waits += 1
            while !(b.reserved + n <= b.capacity || b.reserved == 0)
                wait(b.cond)
            end
        end
        b.reserved += n
        b.peak_reserved = max(b.peak_reserved, b.reserved)
        b.admissions += 1
    finally
        unlock(b.cond)
    end
    n
end

"""
$(TYPEDSIGNATURES)

Return `n` bytes to the budget and wake anything waiting on it.
"""
function release!(b::DeviceBudget, n::Integer)
    n = Int(n)
    n <= 0 && return nothing
    lock(b.cond)
    try
        b.reserved -= n
        notify(b.cond; all = true)
    finally
        unlock(b.cond)
    end
    nothing
end

Base.show(io::IO, b::DeviceBudget) = print(
    io,
    "DeviceBudget(capacity=$(format_bytes(b.capacity)), ",
    "reserved=$(format_bytes(b.reserved)), peak=$(format_bytes(b.peak_reserved)), ",
    "admissions=$(b.admissions), waits=$(b.waits))",
)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Per-transformation record produced by [`sweep_transformations`](@ref).

# Fields
- `index::Int`: position in the transformation list.
- `transformation::LatticeTransformation`: the transformation solved.
- `energy::Float64`: lowest energy this transformation found (`NaN` if it failed).
- `wall_time::Float64`: seconds spent in the solve.
- `truncation::TruncationStats`: weight discarded by the boundary-MPS
  truncations of this solve. `discarded_sum` is the leading-order accumulated
  fidelity loss and `saturated` counts how often the bond dimension (rather than
  the singular-value tolerance) forced a non-negligible discard — together these say
  whether the contraction, as opposed to the search, limited the result.
- `largest_discarded_probability::Float64`: the search-side bound already
  reported by [`Solution`](@ref).
- `calibration::Bool`: whether this was the solo calibration run.
- `error`: the exception if the solve failed, otherwise `nothing`.
"""
struct TransformReport
    index::Int
    transformation::LatticeTransformation
    energy::Float64
    wall_time::Float64
    truncation::TruncationStats
    largest_discarded_probability::Float64
    calibration::Bool
    error::Any
end

"""
$(TYPEDSIGNATURES)

Sweep-level diagnostics produced by [`sweep_transformations`](@ref).

# Fields
- `per_transform::Vector{TransformReport}`: one entry per transformation.
- `reservation::Int`: bytes reserved per concurrent solve. `0` means the governor
  stood down and concurrency was limited only by `max_concurrency` — either
  because `reservation = :none` was requested, or because calibration measured a
  peak of zero (see `calibrated_peak`).
- `calibrated_peak::Int`: peak device memory measured during the solo run. Zero
  when CUDA is unavailable, when calibration was skipped, or when the solve is
  small enough that its allocations stay below the granularity at which the
  driver reports free memory — a solve too small to measure is also too small to
  need rationing, so standing the governor down is the right response.
- `capacity::Int`: usable device memory the governor was allowed to hand out.
- `max_concurrency::Int`: admission limit derived from `capacity / reservation`.
- `peak_reserved::Int`: high-water mark of simultaneously reserved bytes.
- `waits::Int`: how many solves blocked on the budget.
- `wall_time::Float64`: total sweep wall time.
- `calibration_time::Float64`: of which, the solo run.
- `energy_spread::Float64`: best-to-worst energy range across transformations —
  a cheap, oracle-free indicator of whether the contraction is trustworthy. A
  spread that is a sizeable fraction of the energy scale means the eight solves
  disagree and the result should not be believed.
- `consensus::Int`: how many transformations reached (within tolerance) the best
  energy found.
- `failures::Int`: how many transformations threw.
"""
struct SweepReport
    per_transform::Vector{TransformReport}
    reservation::Int
    calibrated_peak::Int
    capacity::Int
    max_concurrency::Int
    peak_reserved::Int
    waits::Int
    wall_time::Float64
    calibration_time::Float64
    energy_spread::Float64
    consensus::Int
    failures::Int
end

"""
$(TYPEDSIGNATURES)

Result of a transformation sweep: the individual solutions, which one was best,
and the diagnostics gathered along the way.

# Fields
- `transformations::Vector{LatticeTransformation}`
- `solutions::Vector{Union{Nothing,Solution}}`: aligned with `transformations`;
  `nothing` for a transformation that failed.
- `best_index::Int`: index of the lowest-energy solution (0 if all failed).
- `report::SweepReport`
"""
struct SweepSolution
    transformations::Vector{LatticeTransformation}
    solutions::Vector{Union{Nothing,Solution}}
    best_index::Int
    report::SweepReport
end

"""
$(TYPEDSIGNATURES)

The lowest-energy [`Solution`](@ref) of a sweep. Throws if every transformation
failed.
"""
function best_solution(s::SweepSolution)
    s.best_index == 0 && error("every transformation in the sweep failed")
    s.solutions[s.best_index]
end

function Base.show(io::IO, s::SweepSolution)
    r = s.report
    print(
        io,
        "SweepSolution($(length(s.transformations)) transformations, ",
        "best=$(s.best_index == 0 ? "none" : string(round(r.per_transform[s.best_index].energy, sigdigits=8))), ",
        "consensus=$(r.consensus)/$(length(s.transformations)), ",
        "concurrency=$(r.max_concurrency), ",
        "reservation=$(format_bytes(r.reservation)))",
    )
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Deterministic per-transformation seeding. The Zipper strategy draws a random
# sketch for its randomized range finder, so without explicit seeding a parallel
# sweep would return results that depend on task scheduling. Julia's and
# CUDA.jl's default RNGs are both task-local, so seeding at the top of each task
# gives every transformation an independent, reproducible stream regardless of
# the order tasks happen to run in.
function _seed_task!(seed::Integer, index::Integer)
    s = hash((UInt64(seed), UInt64(index)))
    Random.seed!(Random.default_rng(), s)
    CUDA.functional() && CUDA.seed!(s % UInt64)
    nothing
end

# Usable device memory to hand out. `available_memory()` after a reclaim is the
# honest number: total capacity would ignore the driver context, other processes,
# and anything the caller already has resident.
function _usable_device_memory(fraction::Real)
    CUDA.functional() || return typemax(Int) ÷ 2
    GC.gc(false)
    CUDA.reclaim()
    max(Int(floor(CUDA.available_memory() * fraction)), 1)
end

# Run `f` while sampling device memory, returning (result, peak_bytes). Two
# samplers feed one tracker; see `DevicePeak`.
function _with_device_peak(f)
    CUDA.functional() || return f(), 0
    GC.gc(false)
    CUDA.reclaim()
    tracker = DevicePeak()
    stop = Threads.Atomic{Bool}(false)
    watcher = if Threads.nthreads() > 1
        Threads.@spawn begin
            while !stop[]
                Threads.atomic_min!(tracker.low, Int(CUDA.available_memory()))
                sleep(0.005)
            end
        end
    else
        nothing
    end
    try
        result = with(DEVICE_PEAK_PROBE => tracker) do
            f()
        end
        return result, device_peak_bytes(tracker)
    finally
        stop[] = true
        watcher === nothing || wait(watcher)
    end
end

# Free whatever the finished solve left resident before the next one is admitted.
# The contraction cache is already released by `low_energy_spectrum`, but those
# buffers sit in CUDA.jl's caching allocator until reclaimed, so the next task
# would see them as unavailable and the byte budget would be fiction.
function _release_device_memory()
    CUDA.functional() || return nothing
    GC.gc(false)
    CUDA.reclaim()
    nothing
end

_energy_of(sol::Solution) = isempty(sol.energies) ? NaN : Float64(first(sol.energies))

# Default admission limit for `concurrency = :auto`.
#
# On a GPU: one, conservatively. Whether fanning the transformations out over a
# single device pays depends on the card. On a consumer GPU it does not — measured
# on an RTX 5080 over 8 transformations (7 interleaved A/B rounds, full GC +
# `CUDA.reclaim()` before every timed section, median of per-round paired ratios),
# the concurrent sweep ran at 0.92x/0.80x (c=2/c=4) on the 3x4x3 case and
# 0.88x/0.89x on 128power: the solves overlap but per-solve time degrades at the
# same rate (utilization ~10%, the limit being serialization in the CUDA
# API/allocator with this solver's many small kernels). A datacenter GPU has the
# headroom to overlap them — on an H100 the same sweeps reach 1.69x/1.44x and
# 1.22x/1.43x (c=2/c=4). Because the common case is the smaller card, `:auto` stays
# at 1 on any GPU; set `concurrency = 2`–`4` explicitly on a large device.
#
# (Interleaving and reclaiming matter more than the effect being measured: a
# naive protocol that timed the serial arm right after a concurrent warm-up
# reported a spurious speed-up on the 5080, purely from leftover pool state
# inflating the baseline by up to 2.7x.)
#
# On CPU there is no such shared bottleneck and concurrency pays, monotonically
# (Xeon Platinum 8462Y+, 5 rounds, same protocol):
#
#   case                            serial   c=1    c=2    c=4    c=8
#   chimera 3x4x3, D=16, SVD         0.04 s  0.88x  1.41x  2.14x  3.09x
#   chimera 128power, D=32, Zipper  16.0 s   0.89x  1.11x  1.31x  1.64x
#
# so use the thread count there and let BLAS threads be divided among the admitted
# solves. (`c=1` is below 1.0 because the driver's fixed overhead is not amortized
# when nothing overlaps.)
function _auto_concurrency(n::Int)
    CUDA.functional() ? 1 : min(n, Threads.nthreads())
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Solve one instance once per lattice transformation, concurrently, under a device
memory budget, and return every solution together with diagnostics.

Replaces the serial `for transform ∈ all_lattice_transformations` loop that the
published examples spell out by hand. The transformations are independent, so
the only thing that limits the fan-out is device memory; this function measures
what one solve costs and admits as many as fit. See the file header for why the
governor is denominated in bytes.

# Arguments
- `build_contractor`: `transformation -> MpsContractor`. Called once per
  transformation, inside the task that will run it, so each solve gets its own
  network, projector workspace, and contraction cache.
- `sparams::SearchParameters`: forwarded to [`low_energy_spectrum`](@ref).

# Keyword arguments
- `transformations = all_lattice_transformations`: which transformations to run.
- `merge_strategy = _ -> no_merge`: `ctr -> strategy`, because the published
  merge strategies close over the contractor (`merge_branches(ctr; ...)`).
- `symmetry::Symbol = :noZ2`: forwarded to [`low_energy_spectrum`](@ref).
- `concurrency = :auto`: cap on simultaneously running solves. `:auto` is **1 on a
  GPU** — a conservative default: on a consumer GPU (e.g. RTX 5080) fanning these
  solves out over one device does not beat the serial loop, but a datacenter GPU
  (e.g. H100) does benefit, so set `concurrency = 2`–`4` explicitly there. On CPU it
  is `min(length(transformations), Threads.nthreads())`. Raise it explicitly for a
  CPU-only run, several devices, or an instance you have measured. The byte budget
  may admit fewer.
- `reservation = :calibrate`: bytes to reserve per solve. `:calibrate` measures
  it from a solo run; pass an integer to skip calibration (useful when you
  already know the figure and want all transformations to start at once), or
  `:none` to disable the governor entirely.
- `reservation_headroom = 1.3`: multiplier on the calibrated peak, covering
  instance-to-instance variation between transformations.
- `vram_fraction = 0.85`: fraction of free device memory the governor may hand
  out, leaving room for fragmentation and the driver context.
- `seed = 1234`: base seed; transformation `i` is seeded with
  `hash((seed, i))`. `nothing` leaves RNG state alone, which makes a `Zipper`
  sweep non-reproducible.
- `blas_threads = :auto`: BLAS threads per solve during the parallel phase.
  `:auto` divides the current setting by the admission limit — several solves
  each calling multi-threaded LAPACK (`qr_fact` falls back to the CPU below its
  shape threshold) otherwise oversubscribe the machine badly.
- `diagnostics = true`: record truncation error. Costs two device reductions per
  truncating factorization.
- `show_progress = false`: per-solve progress bars. Off by default because
  concurrent bars interleave into noise.

# Returns
A [`SweepSolution`](@ref). Use [`best_solution`](@ref) for the winner and
`.report` for the diagnostics.

# Example
```julia
potts_h = potts_hamiltonian(ising_graph(instance); spectrum = full_spectrum,
                            cluster_assignment_rule = super_square_lattice((m, n, t)))
params  = MpsParameters{Float64}(; bond_dim = 16, num_sweeps = 1)

sweep = sweep_transformations(
    t -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{KingSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, t),
        params; onGPU = true, beta = 2.0, graduate_truncation = true,
    ),
    SearchParameters(; max_states = 2^8, cutoff_prob = 1e-4);
    merge_strategy = ctr -> merge_branches(ctr; merge_prob = :none),
)

sol = best_solution(sweep)
sweep.report.energy_spread   # do the transformations agree?
```
"""
function sweep_transformations(
    build_contractor,
    sparams::SearchParameters;
    transformations = all_lattice_transformations,
    merge_strategy = _ -> no_merge,
    symmetry::Symbol = :noZ2,
    concurrency::Union{Symbol,Int} = :auto,
    reservation::Union{Symbol,Int} = :calibrate,
    reservation_headroom::Real = 1.3,
    vram_fraction::Real = 0.85,
    seed::Union{Nothing,Integer} = 1234,
    blas_threads::Union{Symbol,Int} = :auto,
    diagnostics::Bool = true,
    show_progress::Bool = false,
)
    transforms = collect(transformations)
    isempty(transforms) && throw(ArgumentError("no transformations to sweep"))
    0 < vram_fraction <= 1 ||
        throw(ArgumentError("vram_fraction must be in (0, 1], got $vram_fraction"))
    reservation_headroom >= 1 || throw(
        ArgumentError("reservation_headroom must be >= 1, got $reservation_headroom"),
    )

    concurrency isa Int ||
        concurrency === :auto ||
        throw(ArgumentError("concurrency must be an Int or :auto, got $concurrency"))
    reservation isa Int ||
        reservation ∈ (:calibrate, :none) ||
        throw(
            ArgumentError("reservation must be an Int, :calibrate or :none, got $reservation"),
        )

    n = length(transforms)
    solutions = Vector{Union{Nothing,Solution}}(nothing, n)
    reports = Vector{Union{Nothing,TransformReport}}(nothing, n)
    t_start = time()

    workers = concurrency === :auto ? _auto_concurrency(n) : Int(concurrency)
    workers = clamp(workers, 1, n)
    if concurrency === :auto && Threads.nthreads() == 1 && n > 1
        @info """
        Sweeping $n transformations serially: this session has one Julia thread.
        Start Julia with `-t auto` (or `JULIA_NUM_THREADS`) to run them concurrently.
        """
    end

    # One solve, instrumented. Runs inside whatever scope the caller sets up.
    function run_one(index::Int, is_calibration::Bool)
        transform = transforms[index]
        log = diagnostics ? TruncationLog() : nothing
        t0 = time()
        try
            seed === nothing || _seed_task!(seed, index)
            ctr = build_contractor(transform)
            sol, _ = with(TRUNCATION_LOG => log) do
                low_energy_spectrum(
                    ctr,
                    sparams,
                    merge_strategy(ctr),
                    symmetry;
                    show_progress = show_progress,
                )
            end
            solutions[index] = sol
            reports[index] = TransformReport(
                index,
                transform,
                _energy_of(sol),
                time() - t0,
                truncation_stats(log),
                Float64(sol.largest_discarded_probability),
                is_calibration,
                nothing,
            )
        catch err
            err isa InterruptException && rethrow()
            @error "transformation $index failed" exception =
                (err, catch_backtrace())
            reports[index] = TransformReport(
                index,
                transform,
                NaN,
                time() - t0,
                truncation_stats(log),
                NaN,
                is_calibration,
                err,
            )
        end
        nothing
    end

    # ---- phase 1: calibration -------------------------------------------
    capacity = reservation === :none ? 0 : _usable_device_memory(vram_fraction)
    per_solve = 0
    peak = 0
    t_cal = 0.0

    if reservation === :calibrate
        # Provisional budget: the slice this solve would get in the parallel
        # phase, so the measured peak reflects the regime we will actually run
        # in rather than a solo run that batched against all of free memory.
        provisional = max(capacity ÷ workers, 1)
        cal = time()
        _, peak = _with_device_peak() do
            with(DEVICE_MEMORY_BUDGET => provisional) do
                # In a task, not inline: `_seed_task!` seeds the *current* task's
                # RNG, and doing that on the caller's task would silently reseed
                # the enclosing script's global RNG. Scoped values are inherited
                # by spawned tasks, so the budget still applies.
                fetch(Threads.@spawn run_one(1, true))
            end
        end
        t_cal = time() - cal
        # A zero peak means the solve's allocations never moved the driver's
        # free-memory figure. Reserving zero stands the governor down, which is
        # the correct response: such a solve cannot exhaust the device however
        # many run at once.
        per_solve = peak > 0 ? Int(ceil(peak * reservation_headroom)) : 0
    elseif reservation isa Int
        per_solve = reservation
    end

    started_at = reservation === :calibrate ? 2 : 1
    max_conc = if per_solve > 0
        clamp(capacity ÷ per_solve, 1, workers)
    else
        workers
    end

    budget = per_solve > 0 ? DeviceBudget(capacity) : nothing

    # ---- phase 2: parallel remainder ------------------------------------
    blas_before = BLAS.get_num_threads()
    if started_at <= n
        try
            if max_conc > 1
                per_task = if blas_threads === :auto
                    max(1, blas_before ÷ max_conc)
                elseif blas_threads isa Int
                    max(1, blas_threads)
                else
                    blas_before
                end
                BLAS.set_num_threads(per_task)
            end
            _release_device_memory()

            # Two independent limits. The semaphore honours the caller's
            # `concurrency` cap on running solves; the byte budget honours the
            # device. Gating the semaphore at `max_conc` too would make the budget
            # decorative — it could never block, and `waits` would always read
            # zero however tight memory was.
            gate = Base.Semaphore(workers)
            # Reclaiming after every solve costs a GC pass plus a device-pool
            # release — measurable (~30% of a short sweep) and pointless unless a
            # queued solve is waiting for those bytes to come back. That needs
            # both actual concurrency and more solves than fit at once; with
            # `max_conc == 1` nothing runs alongside anything, and CUDA.jl reuses
            # its pool across successive solves without help.
            reclaim_between =
                per_solve > 0 && max_conc > 1 && (n - started_at + 1) > max_conc
            tasks = map(started_at:n) do index
                Threads.@spawn begin
                    Base.acquire(gate)
                    held = 0
                    try
                        held = budget === nothing ? 0 : reserve!(budget, per_solve)
                        if per_solve > 0
                            with(DEVICE_MEMORY_BUDGET => per_solve) do
                                run_one(index, false)
                            end
                        else
                            run_one(index, false)
                        end
                    finally
                        reclaim_between && _release_device_memory()
                        budget === nothing || release!(budget, held)
                        Base.release(gate)
                    end
                end
            end
            foreach(wait, tasks)
            # Return whatever the last solves left resident, once, rather than
            # after each of them.
            _release_device_memory()
        finally
            BLAS.set_num_threads(blas_before)
        end
    end

    # ---- assemble --------------------------------------------------------
    final = TransformReport[r for r ∈ reports if r !== nothing]
    sort!(final, by = r -> r.index)
    energies = [r.energy for r ∈ final if !isnan(r.energy)]
    best_index = 0
    spread = NaN
    consensus = 0
    if !isempty(energies)
        best_e = minimum(energies)
        for r ∈ final
            if !isnan(r.energy) && r.energy == best_e
                best_index = r.index
                break
            end
        end
        spread = maximum(energies) - best_e
        # Count transformations that agree with the best energy to within a
        # relative tolerance; how many independent contraction orders found the
        # same minimum is the sweep's oracle-free confidence signal.
        scale = max(abs(best_e), eps())
        consensus = count(e -> abs(e - best_e) <= 1e-8 * scale, energies)
    end

    report = SweepReport(
        final,
        per_solve,
        peak,
        capacity,
        max_conc,
        budget === nothing ? 0 : budget.peak_reserved,
        budget === nothing ? 0 : budget.waits,
        time() - t_start,
        t_cal,
        spread,
        consensus,
        count(r -> r.error !== nothing, final),
    )
    SweepSolution(transforms, solutions, best_index, report)
end
