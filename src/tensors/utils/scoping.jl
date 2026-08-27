# scoping.jl: task-scoped execution context for a single solve.
#
# Two quantities need to be visible to every kernel in a contraction without
# being threaded through ~40 function signatures, and need to differ between
# concurrently running solves (one per lattice transformation):
#
#   * the slice of device memory this solve is allowed to use, so that kernel
#     batch budgets are sized against a reservation the caller can honour
#     instead of against whatever happens to be free at the moment;
#   * an accumulator for truncation error, so a heuristic contraction can report
#     how much weight it threw away.
#
# `ScopedValue` (Julia >= 1.11) is exactly the right tool: dynamically scoped,
# task-local, and inherited by tasks spawned inside the scope. Both default to
# "inactive", so nothing changes for callers that do not opt in.

export DEVICE_MEMORY_BUDGET,
    DEVICE_PEAK_PROBE,
    TRUNCATION_LOG,
    DevicePeak,
    TruncationLog,
    TruncationStats,
    device_memory_budget,
    device_peak_bytes,
    probe_device_peak!,
    record_truncation!,
    truncation_stats

using Base.ScopedValues: ScopedValue

"""
Device memory (in bytes) that the current task may use for kernel
intermediates, or `0` for "unrestricted — infer from free device memory".

Set by the parallel sweep driver so that concurrent solves size their kernel
batches against disjoint slices of VRAM rather than each measuring the same
shared free pool. See [`kernel_batch_size`](@ref).
"""
const DEVICE_MEMORY_BUDGET = ScopedValue{Int}(0)

"""
$(TYPEDSIGNATURES)

The device-memory budget in force for the current task, in bytes; `0` means no
explicit budget is set and the caller should fall back to querying the device.
"""
device_memory_budget() = DEVICE_MEMORY_BUDGET[]

"""
Tracks how much device memory a solve actually used, as a high-water mark over
samples of `CUDA.available_memory()` relative to a baseline taken at
construction.

CUDA.jl 5.x allocates through its own caching allocator rather than the CUDA
stream-ordered pool, so the driver's `MEMPOOL_ATTR_USED_MEM_HIGH` watermark
stays at zero and cannot be used. Sampling free memory does track the caching
allocator (verified), so peak usage is reconstructed from samples taken by two
independent samplers:

  * a watchdog task polling on a timer, which catches transients inside a row
    but needs a second thread to be scheduled reliably;
  * `probe_device_peak!` calls placed at natural synchronization points in the
    solve (row boundaries), which are deterministic and work even in a
    single-threaded session.

Both feed the same tracker, so the reported peak is the max over both.

Only meaningful while the tracked solve runs alone on the device — concurrent
work would be attributed to it. This is why the sweep driver calibrates on a
solo run before admitting any concurrency.
"""
struct DevicePeak
    base::Int
    low::Threads.Atomic{Int}
end

function DevicePeak()
    base = CUDA.functional() ? Int(CUDA.available_memory()) : 0
    DevicePeak(base, Threads.Atomic{Int}(base))
end

"""
$(TYPEDSIGNATURES)

Peak device memory in bytes attributed to the tracked region so far.
"""
device_peak_bytes(p::DevicePeak) = max(0, p.base - p.low[])

"""
$(TYPEDSIGNATURES)

Take one device-memory sample for the peak tracker installed in the current
task, if any. A no-op when no tracker is in scope or CUDA is unavailable, so it
is safe to call unconditionally from hot-ish code such as a row transition.
"""
function probe_device_peak!()
    p = DEVICE_PEAK_PROBE[]
    p === nothing && return nothing
    CUDA.functional() || return nothing
    Threads.atomic_min!(p.low, Int(CUDA.available_memory()))
    nothing
end

"""
Peak-memory tracker installed for the current task, or `nothing` when device
peak usage is not being measured (the default).
"""
const DEVICE_PEAK_PROBE = ScopedValue{Union{Nothing,DevicePeak}}(nothing)

"""
Relative discarded weight (‖Σ_dropped‖² / ‖Σ‖²) below which a bond-capped
truncation is not counted as `saturated`: the dropped singular values are
deliberately ignored, so no change from raising the bond dimension is expected
at the reported precision. Sits above the ~1e-14 weights observed in a reference
contraction and below the smallest weight a genuinely bond-limited
truncation has been observed to drop (~1e-10 on the shipped 128-spin fixture).
"""
const NEGLIGIBLE_DISCARD = 1e-12

"""
Running tally of the weight discarded by truncating factorizations.

Every truncating `svd_fact` (and therefore every `qr_fact`/`rq_fact`/
`canonise_truncate!`/zipper call that truncates) adds one entry while a log is
installed in the current task's scope via [`TRUNCATION_LOG`](@ref).

# Fields
- `count::Int`: number of truncating factorizations recorded.
- `discarded_sum::Float64`: Σᵢ εᵢ, where εᵢ is the relative discarded weight
  (‖Σ_dropped‖² / ‖Σ‖²) of factorization `i`. For small εᵢ this is the leading
  term of the accumulated fidelity loss of the contraction, so it is the natural
  single-number error proxy for a boundary-MPS sweep.

  Two caveats on reading it. It is a *sum* over every truncation in the solve, so
  once it approaches or exceeds 1 the linearisation behind that interpretation no
  longer holds and the value only says "this contraction is untrustworthy" — a
  bond-4 solve of a 2500-spin instance reaches Σε ≈ 1.3 over 2162 truncations. And
  it bounds what the *contraction* discarded, which is not the same as how good a
  state the subsequent search found: it is non-monotone in β, and on one measured
  family the β values giving the best energies carried among the lowest discarded
  weight. Use it to judge the contraction, not to rank answers.
- `discarded_max::Float64`: maxᵢ εᵢ — flags a single pathological truncation
  that a sum over many benign ones would hide.
- `saturated::Int`: how many factorizations hit the bond-dimension bound (rather
  than dropping only numerically negligible singular values, per
  [`NEGLIGIBLE_DISCARD`](@ref)). If this is zero, the bond dimension was never
  the binding constraint.
- `dims_kept::Int`, `dims_offered::Int`: retained vs. available singular values,
  summed over all recorded factorizations.

Counters are monotone, so a caller can [`truncation_stats`](@ref) before and
after a phase and subtract to attribute error to that phase.
"""
mutable struct TruncationLog
    count::Int
    discarded_sum::Float64
    discarded_max::Float64
    saturated::Int
    dims_kept::Int
    dims_offered::Int
    lock::ReentrantLock
end

TruncationLog() = TruncationLog(0, 0.0, 0.0, 0, 0, 0, ReentrantLock())

"""
Truncation log installed for the current task, or `nothing` when truncation
error is not being recorded (the default — recording costs two extra reductions
per factorization, so it is opt-in).
"""
const TRUNCATION_LOG = ScopedValue{Union{Nothing,TruncationLog}}(nothing)

"""
Immutable snapshot of a [`TruncationLog`](@ref). Supports `-` so that two
snapshots delimit a phase of the contraction.
"""
struct TruncationStats
    count::Int
    discarded_sum::Float64
    discarded_max::Float64
    saturated::Int
    dims_kept::Int
    dims_offered::Int
end

TruncationStats() = TruncationStats(0, 0.0, 0.0, 0, 0, 0)

"""
$(TYPEDSIGNATURES)

Snapshot the truncation log installed in the current task; returns an empty
`TruncationStats` when no log is installed.
"""
function truncation_stats(log::Union{Nothing,TruncationLog} = TRUNCATION_LOG[])
    log === nothing && return TruncationStats()
    @lock log.lock TruncationStats(
        log.count,
        log.discarded_sum,
        log.discarded_max,
        log.saturated,
        log.dims_kept,
        log.dims_offered,
    )
end

# Difference of two snapshots: counters are monotone, so subtraction attributes
# error to the interval between them. `discarded_max` is not additive; the
# difference reports the later maximum, which is the best available bound on the
# interval without keeping every sample.
function Base.:-(a::TruncationStats, b::TruncationStats)
    TruncationStats(
        a.count - b.count,
        a.discarded_sum - b.discarded_sum,
        a.count == b.count ? 0.0 : a.discarded_max,
        a.saturated - b.saturated,
        a.dims_kept - b.dims_kept,
        a.dims_offered - b.dims_offered,
    )
end

"""
$(TYPEDSIGNATURES)

Zero a truncation log in place, so it can be reused for the next phase.
"""
function Base.empty!(log::TruncationLog)
    @lock log.lock begin
        log.count = 0
        log.discarded_sum = 0.0
        log.discarded_max = 0.0
        log.saturated = 0
        log.dims_kept = 0
        log.dims_offered = 0
    end
    log
end

"""
$(TYPEDSIGNATURES)

Record one truncating factorization that kept `kept` of `offered` singular
values, discarding relative weight `discarded` (‖Σ_dropped‖²/‖Σ‖²). `saturated`
marks a truncation forced by the bond-dimension bound rather than by the
singular-value tolerance, and that discarded more than numerically negligible
weight (callers combine the bond-bound test with [`NEGLIGIBLE_DISCARD`](@ref)).

A no-op when no [`TRUNCATION_LOG`](@ref) is installed in the current task.
"""
function record_truncation!(
    discarded::Real,
    kept::Integer,
    offered::Integer,
    saturated::Bool,
)
    log = TRUNCATION_LOG[]
    log === nothing && return nothing
    ε = Float64(discarded)
    # Guard against round-off producing a tiny negative discarded weight.
    ε = ε < 0 ? 0.0 : ε
    @lock log.lock begin
        log.count += 1
        log.discarded_sum += ε
        log.discarded_max = max(log.discarded_max, ε)
        log.saturated += saturated
        log.dims_kept += kept
        log.dims_offered += offered
    end
    nothing
end

Base.show(io::IO, s::TruncationStats) = print(
    io,
    "TruncationStats(count=$(s.count), Σε=$(round(s.discarded_sum, sigdigits=4)), ",
    "maxε=$(round(s.discarded_max, sigdigits=4)), saturated=$(s.saturated), ",
    "dims=$(s.dims_kept)/$(s.dims_offered))",
)
