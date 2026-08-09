```@meta
CurrentModule = SpinGlassPEPS
```

# Concurrent sweeps and error control

The search sweeps variables in a fixed order tied to the contraction order, so
rotating and reflecting the lattice starts the sweep from a different corner and
materially stabilizes the result. The standard protocol therefore solves the same
instance once per element of `all_lattice_transformations` and keeps the best
outcome.

Those solves are independent. Since the contraction caches became
contractor-owned there is nothing to stop them running at once, and
[`sweep_transformations`](@ref) does exactly that.

```julia
potts_h = potts_hamiltonian(
    ising_graph(instance);
    spectrum = full_spectrum,
    cluster_assignment_rule = super_square_lattice((m, n, t)),
)
params = MpsParameters{Float64}(; bond_dim = 16, num_sweeps = 1)

sweep = sweep_transformations(
    transform -> MpsContractor(
        SVDTruncate,
        PEPSNetwork{KingSingleNode{GaugesEnergy},Dense,Float64}(m, n, potts_h, transform),
        params;
        onGPU = true, beta = 2.0, graduate_truncation = true,
    ),
    SearchParameters(; max_states = 2^8, cutoff_prob = 1e-4);
    merge_strategy = ctr -> merge_branches(ctr; merge_prob = :none),
)

sol = best_solution(sweep)
```

`build_contractor` is called once per transformation *inside the task that will
run it*, so each solve gets its own network, projector workspace, and contraction
cache. Julia must be started with more than one thread (`julia -t auto`) for the
solves to overlap; with a single thread the sweep runs serially and says so.

## How much does concurrency actually buy? On CPU, up to ~1.8×; on one GPU, nothing

The answer depends entirely on the device, so both halves are given below. Same
protocol throughout: all eight transformations, interleaved A/B rounds with a full
`GC.gc` (plus `CUDA.reclaim()` on the device) before every timed section, reported
as the median of the per-round paired ratios against the serial loop.

### On CPU it scales

20 cores, BLAS on 12 threads, `CUDA_VISIBLE_DEVICES=""`, 5 rounds:

| case | serial | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|---|
| chimera 3×4×3, D=16, `SVDTruncate` | 0.07 s | 0.52× | 0.87× | 1.32× | **1.76×** |
| chimera 128power, D=32, `Zipper` | 4.16 s | 0.97× | 1.37× | 1.37× | **1.39×** |

Monotonic in concurrency, so `:auto` uses `min(n, Threads.nthreads())` here. `c=1`
is below 1.0 because the driver's fixed overhead (calibration, bookkeeping) is not
amortized when there is nothing to overlap — it costs most on the 70 ms case.

### On one GPU it does not

RTX 5080, 7 rounds:

| case | serial (median) | sweep, c=2 | sweep, c=4 |
|---|---|---|---|
| chimera 3×4×3, D=16, `SVDTruncate` | 3.97 s | 0.92× | 0.80× |
| chimera 128power, D=32, `Zipper` | 14.30 s | 0.88× | 0.89× |

No concurrency level beats the serial loop. The solves *do* overlap — 5.4× at
eight-way — but per-solve time degrades at the same rate, so total work is
conserved and the driver's own overhead makes it a net loss.

The device is not the constraint: GPU utilization stays near 10% throughout, and
holding concurrency at 8 while varying only the BLAS thread policy moves the total
by under 10%. What saturates is serialization inside the CUDA API and allocator,
which this solver provokes because its kernels are small and numerous. Recovering
that idle device would mean batching across transformations *inside* the kernels
so it sees one large operation instead of eight sets of small ones — not more
tasks. This function does not attempt that.

`concurrency = :auto` is therefore **1 on a GPU**. Raise it explicitly for a
CPU-only run, for several devices, or for an instance you have measured yourself.

!!! note "Measuring this is harder than it looks"
    A naive protocol — warm up, time the serial arm, then time the concurrent one —
    reported 1.68× and 1.06× for these same two cases. That was entirely an
    artifact: timing the serial arm straight after a concurrent warm-up leaves the
    CUDA pool in a state that inflated the baseline by up to 2.7×. Interleave the
    arms and reclaim before every timed section, and report paired per-round
    ratios rather than a min over separate batches.

!!! warning "Below ~2000 spins, this solver is faster on the CPU"
    Compare the serial columns above: 0.07 s versus 3.97 s, and 4.16 s versus
    14.30 s. Solving identical configurations on each device across three instance
    sizes and four bond dimensions puts the crossover well up the size range
    (CPU/GPU wall-clock ratio; below 1 means the host is faster):

    | spins | D=8 | D=16 | D=32 | D=64 |
    |---|---|---|---|---|
    | 36 (dense) | 0.02 | 0.02 | 0.02 | 0.02 |
    | 128 (dense) | 0.15 | 0.38 | 0.48 | 0.56 |
    | 2048 (dense) | 0.43 | 0.68 | **1.03** | — |
    | 2048 (sparse) | — | 0.71 | **1.17** | — |

    The device wins only once both the instance and the bond dimension are large.
    `MpsContractor` defaults to `onGPU = true`, which suits the D-Wave-scale sparse
    regime the package targets and is the wrong default for exploratory work —
    measure before assuming. Energies agree in every cell.

So: on CPU the sweep is a real speed-up and worth turning up; on one GPU it is
not, and its value there is everything other than throughput — one call instead of
a hand-written loop, a device-memory budget that keeps concurrency from exhausting
the card where concurrency does pay, deterministic per-transformation seeding (a
`Zipper` sweep was not reproducible under concurrency before), failure isolation,
and the cross-transformation agreement diagnostics below.

## Why the governor counts bytes

The constraint on fanning out eight solves is not CPU cores, it is device memory.
A single bond-32 solve on a Pegasus-scale instance can hold several GiB, so a
fixed eight-way fan-out reliably exhausts a consumer card, while a small bond-16
Chimera instance would happily run all eight. The number of solves that fit is a
property of the *instance*, not of the machine — so concurrency is gated by a byte
budget:

1. One transformation is solved alone while its peak device usage is measured.
   This is not overhead: its solution counts toward the sweep, and it absorbs the
   compilation that eight concurrent solves would otherwise duplicate under
   contention.
2. That measured peak, times `reservation_headroom`, becomes the per-solve
   reservation. As many reservations as fit in `vram_fraction` of free device
   memory are admitted at once; the rest queue.
3. Each admitted solve runs with its reservation installed as
   [`DEVICE_MEMORY_BUDGET`](@ref), so [`kernel_batch_size`](@ref) sizes kernel
   intermediates against *that slice*.

Step 3 is what makes the reservation real rather than advisory. Without it every
concurrently running solve would measure the same free pool and size its batches
as though it owned all of it.

A reservation larger than the whole budget is admitted alone rather than
deadlocking, so a budget that turns out to be too small degrades to serial
execution instead of hanging. Pass `reservation = <bytes>` to skip calibration
when you already know the figure, or `reservation = :none` to disable the
governor.

```julia
r = sweep.report
r.calibrated_peak    # bytes measured during the solo run
r.reservation        # bytes reserved per concurrent solve
r.max_concurrency    # how many were admitted at once
r.waits              # how many solves blocked -> the sweep was VRAM-bound
```

## Reproducibility

The `Zipper` strategy draws a random sketch for its randomized range finder, so
without explicit seeding a concurrent sweep would return results that depend on
task scheduling. Each transformation is seeded from `hash((seed, index))`; both
Julia's and CUDA.jl's default RNGs are task-local, so every transformation gets
an independent, reproducible stream whatever order the tasks run in. Pass
`seed = nothing` to opt out.

A bare `low_energy_spectrum` does no seeding — that is the caller's job. The
sketch is drawn from the global RNG, so seed it yourself if you need results
that are bit-identical across sessions:

```julia
using Random
Random.seed!(1234)
sol, _ = low_energy_spectrum(ctr, search_params)
```

In practice the range finder is not fragile: on a 2500-spin instance, six
different seeds and five BLAS thread counts all returned the same energy to the
last bit. Seed anyway when a number is going to be published, since the
guarantee costs nothing.

## Error control

A heuristic contraction is only as trustworthy as the weight it keeps. Truncating
factorizations report what they discard into a task-scoped
[`TruncationLog`](@ref); `sweep_transformations` installs one per transformation
and reports the result.

```julia
t = sweep.report.per_transform[1].truncation
t.discarded_sum   # Σᵢ εᵢ — leading-order accumulated fidelity loss
t.discarded_max   # the worst single truncation
t.saturated       # how many truncations the bond dimension forced
t.dims_kept, t.dims_offered
```

`saturated == 0` means the bond dimension was never the binding constraint — the
truncations dropped only numerically negligible singular values, so raising
`bond_dim` will not help. A large `discarded_sum` means the opposite.

Two caveats on `discarded_sum`. It sums over every truncation in the solve, so once
it approaches 1 the "accumulated fidelity loss" reading breaks down and the number
only tells you the contraction is untrustworthy — a bond-4 solve of a 2500-spin
instance reaches ≈ 1.3 over 2162 truncations. And it describes the contraction, not
the answer: see the β warning below.

Two sweep-level numbers need no oracle to interpret:

```julia
sweep.report.energy_spread   # best-to-worst energy across transformations
sweep.report.consensus       # how many transformations reached the best energy
```

Eight independent contraction orders that agree is meaningful evidence; a spread
that is a sizeable fraction of the energy scale means they do not, and the result
should not be believed regardless of how good the best energy looks.

To record truncation error for a single solve, install a log yourself:

```julia
using Base.ScopedValues: with

log = TruncationLog()
with(TRUNCATION_LOG => log) do
    low_energy_spectrum(ctr, search_params; show_progress = false)
end
truncation_stats(log)
```

Recording costs two device reductions per truncating factorization, so it is
opt-in; with no log installed it costs nothing.

## Choosing β

β is the solver's most consequential parameter. It sets how sharply the
represented Boltzmann distribution concentrates on low-energy states: too small
and the conditional probabilities the search branches on say little about the
ground state. The optimal value depends on the instance, which the original
documentation conceded without offering a way to find it.

[`beta_ladder`](@ref) walks an increasing schedule, and each step reuses the
previous step's boundary MPS as the starting point for variational compression
rather than rebuilding `W * ψ` exactly and truncating it:

```julia
ladder = beta_ladder(ctr, [2.0, 3.0, 4.0, 6.0], search_params)
sol = selected_solution(ladder)
[(s.beta, s.energy, s.truncation.discarded_sum) for s ∈ ladder.steps]
```

Selection is on energy. Each rung also reports its discarded weight, so a scan
yields evidence about both the answer and the contraction behind it.

!!! warning "Discarded weight does not select β"
    It is tempting to read `max_discarded` as a quality criterion. It is not, and
    on at least one family it points the wrong way. Ten 2500-spin square-lattice
    instances, bond 8, 500 states:

    | β | 2 | 3 | 4 | 6 | 8 |
    |---|---|---|---|---|---|
    | Σε (median) | 2.3e-4 | 2.2e-3 | 3.4e-3 | 7.3e-4 | 1.4e-4 |
    | energy error (median) | 7.8e-4 | 2.1e-4 | 6.2e-5 | **0** | **0** |

    Σε is *not* monotone in β. It rises while the distribution is still
    sharpening — more structure for the boundary MPS to carry — then falls once
    the distribution concentrates enough to sit close to a product state, which
    truncates easily. Solution quality keeps improving throughout, so here the
    best βs carry among the *lowest* discarded weight and a Σε guard would have
    rejected the good rungs.

    Σε answers "how much of the distribution did the contraction throw away?".
    That is worth knowing, and a large value is a real warning. It is not a proxy
    for whether the search then found a good state.

`stop_when_untrusted` follows from the same caveat: use it to bound cost when a
rung blows up, not to conclude that higher β cannot help — a later rung may well
come back under the guard.

The contractor is mutated in place at every step, so pass one the call may own.

### Warm starting and the error guard do not mix

Use `max_discarded` with `warm_start = false`. The two measure different things:

* a **cold** build forms `W * ψ` exactly, then truncates it — the weight it drops
  is a genuine discarded weight, and `svd_fact` records it;
* a **warm** start optimizes the previous β's MPS *within* a fixed bond dimension
  and never performs a truncating factorization, so it reports ~zero discarded
  weight however accurate it actually is. Its error is a variational optimization
  gap, which the truncation log does not measure.

Measured on the 2048power instance over β = 1.5, 2.25, 3.0, the cold ladder
reports `Σε = [3.2e-3, 1.7e-4, 1.9e-5]` while the warm one reports
`[3.2e-3, 0, 0]` for identical energies — the zeros are an artifact of where the
truncation happens, not better accuracy. Setting both warns for this reason.

So: warm start to go faster (~15% per warmed rung on that instance, where
boundary-MPS construction dominates), cold to audit the contraction.

## Examples

Three runnable scripts in `examples/`, in increasing order of scale:

- `beta_ladder.jl` — 18 spins, runs in seconds. Error control and the β ladder,
  annotated inline; the place to start.
- `concurrent_sweep.jl` — 128 spins. The transformation sweep with the
  device-memory governor.
- `square_50x50.jl` — 2500 spins, the size used by the article's figures. All
  three features, including the agreement diagnostics on a hard instance.

## API

```@docs
sweep_transformations
best_solution
SweepSolution
SweepReport
TransformReport
DeviceBudget
SpinGlassPEPS.SpinGlassEngine.reserve!
SpinGlassPEPS.SpinGlassEngine.release!
DEVICE_MEMORY_BUDGET
beta_ladder
set_beta!
selected_solution
BetaLadderSolution
BetaStepReport
TruncationLog
TruncationStats
truncation_stats
TRUNCATION_LOG
```
