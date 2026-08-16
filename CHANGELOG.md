# Changelog

All notable changes to `SpinGlassPEPS.jl` are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [2.0.0]

### Added

- **Concurrent sweep over lattice transformations, with a device-memory
  governor.** `sweep_transformations(build_contractor, search_params; ...)`
  replaces the serial `for transform ∈ all_lattice_transformations` loop that
  callers previously wrote by hand (and that the published examples spell out).
  The transformations are independent, so the only thing bounding the fan-out is
  VRAM; concurrency is therefore gated by a **byte budget** (`DeviceBudget`)
  rather than a task count:
  - one transformation is solved alone while its peak device usage is measured;
    that run's solution counts, and it absorbs the JIT compilation that eight
    concurrent solves would otherwise duplicate under contention;
  - the measured peak (× `reservation_headroom`) becomes the per-solve
    reservation, and as many reservations as fit in `vram_fraction` of free
    device memory are admitted at once. An oversized request runs alone rather
    than deadlocking, so a bad budget degrades to serial execution;
  - each admitted solve runs with `DEVICE_MEMORY_BUDGET` set to its reservation,
    so `kernel_batch_size` sizes kernel intermediates against that slice. Without
    this the reservation would be advisory: every concurrent solve would measure
    the same free pool and batch as if it owned all of it.
  Transformations are seeded deterministically per index, so a `Zipper` sweep
  (which draws a random sketch) is reproducible regardless of task scheduling.
  BLAS threads are divided among admitted solves for the concurrent phase, since
  `qr_fact` falls back to CPU LAPACK below its shape threshold. A transformation
  that throws is recorded and does not lose the others.

  **Concurrency pays on both devices.** Measured over all eight transformations
  with interleaved A/B rounds and a full GC (plus `CUDA.reclaim()` on the device)
  before every timed section, as the median of paired per-round ratios against the
  serial loop. On CPU (Xeon Platinum 8462Y+) it is monotonic in concurrency,
  reaching **3.09×** on a small `SVDTruncate` case and **1.64×** on a bond-32
  `Zipper` case at eight-way. On an H100 GPU it also helps, reaching **1.69×/1.44×**
  (c=2/c=4) on the `SVDTruncate` case and **1.22×/1.43×** on the bond-32 `Zipper`
  case; a single admitted solve (c=1) is a slight net loss because the driver's
  fixed overhead is not amortized when nothing overlaps, so the gain begins at c≥2.
  `concurrency = :auto` is **1 on a GPU** — a conservative default: on a consumer
  GPU (e.g. RTX 5080) the sweep does not beat the serial loop (the limit is
  serialization in the CUDA API and allocator, with this solver's many small
  kernels), while a datacenter GPU has the headroom to overlap the solves, so set
  `concurrency = 2`–`4` explicitly on a large device. On CPU it is
  `min(n, nthreads)`.

  Beyond throughput, the sweep replaces a hand-written loop with one call, adds a
  VRAM budget, gives deterministic per-transformation seeding (a `Zipper` sweep was
  not reproducible under concurrency at all before), isolates failures, and reports
  the agreement diagnostics.

- **Inverse-temperature ladders with warm-started boundary MPS.**
  `beta_ladder(ctr, betas, search_params; ...)` walks an increasing β schedule,
  re-targeting the contractor at each step (`set_beta!`). Each step reuses the
  previous step's bottom boundary MPS as the starting point for variational
  compression instead of rebuilding `W * ψ` exactly and truncating it
  (`low_energy_spectrum(...; retain_mps = true)`; guesses are snapshotted to host
  memory during preprocessing, since the search consumes each row's MPS as it
  absorbs it). Warm-started and independent ladders return identical energies
  under both `SVDTruncate` and `Zipper`.

  Measured on the 2048power instance (β = 1.5, 2.25, 3.0, bond 16, `Zipper`,
  `Sparse`), where boundary-MPS construction dominates: **~15% faster per warmed
  rung** (48.2 s → 41.7 s, 49.3 s → 41.2 s), 9.6% over the whole ladder, identical
  energies. On a search-dominated instance the gain is only 4–6%, so the benefit
  tracks how much of the solve is spent building boundary MPS.

  **`max_discarded` requires `warm_start = false`.** A cold build truncates an
  exact `W * ψ`, so its discarded weight is recorded; a warm start optimizes within
  a fixed bond dimension, never performs a truncating factorization, and therefore
  reports ~zero discarded weight regardless of accuracy (its error is an
  optimization gap the truncation log does not measure). On the instance above the
  cold ladder reports `Σε = [3.2e-3, 1.7e-4, 1.9e-5]` against the warm ladder's
  `[3.2e-3, 0, 0]` for the same energies. Setting both now warns.
- **Contraction error control.** Truncating factorizations now report the weight
  they discard into a task-scoped `TruncationLog` (`TRUNCATION_LOG`), so a solve
  can say how much of the distribution its contraction threw away instead of
  offering only `largest_discarded_probability`. `TruncationStats` records Σε, max
  ε, how many truncations were bond-dimension-limited rather than
  tolerance-limited, and retained-vs-offered dimensions; snapshots subtract, so a
  caller can attribute error to a phase. Recording is opt-in (two device
  reductions per factorization) and costs nothing when no log is installed.
  `beta_ladder` uses it as an error guard (`max_discarded`), and
  `sweep_transformations` reports per-transformation error plus `energy_spread`
  and `consensus` — an oracle-free indication of whether the eight contraction
  orders agree.

  **Σε does not rank solution quality, and `beta_ladder` selects on energy.** The
  guard only excludes rungs whose contraction is untrustworthy. Measured over ten
  2500-spin instances, energy error falls monotonically with β — reaching the best
  energy found in 9 of 10 instances at β = 6 — while Σε *peaks* at β = 4 (3.4e-3)
  and then falls (7.3e-4 at β = 6, 1.4e-4 at β = 8), because a sharply peaked
  Boltzmann distribution carries less entanglement and truncates more easily. The
  best β values therefore carried among the *lowest* discarded weight, and a guard
  on Σε alone would have rejected them. Σε also being a sum, it is readable as an
  accumulated fidelity loss only while it stays well below 1: a bond-4 solve of a
  2500-spin instance reaches Σε ≈ 1.3 over 2162 truncations, yet returns a *better*
  energy than a bond-32 solve at Σε ≈ 1.7e-11. It bounds what the contraction
  discarded, not whether the search found a good state.

- Explicit, documented `AbstractGeometry` protocol with a conformance testset.
- Device-native QR/RQ via CUSOLVER for GPU-resident matrices, gated by a measured
  size/shape rule; batched GPU kernel for the King-geometry conditional
  probability; device-aware kernel batch budgets (scale with element type and
  available device memory).
- Cached merged projectors and projector indicator matrices in `PoolOfProjectors`.
- Grouped test suite driven by `SPINGLASS_TEST_GROUP`
  (`tensors`/`networks`/`exhaustive`/`engine`/`umbrella`); memory-heavy droplet
  tests skip on GPUs below a memory threshold.
- Unified documentation: one Documenter build over all modules, replacing the
  previous per-package aggregation.
- Benchmark harness (`benchmark/`) with committed reference baselines.
- **Benchmark instances at article scale are now shipped.** The 100 2500-spin
  (50x50) square-lattice instances used by the original article's figures live in
  `benchmark/instances/square_50x50/`. The listing published in the 2025 article
  resolved them from a directory that did not contain them, so its figures could
  not be reproduced from a clean checkout.
- **Three runnable examples**, covering the additions above at three scales:
  `examples/beta_ladder.jl` (18 spins, seconds — error control and the beta
  ladder, annotated inline), `examples/concurrent_sweep.jl` (128 spins — the
  transformation sweep and its memory governor), and
  `examples/square_50x50.jl` (2500 spins — all three at the size used by the
  article's figures, including the agreement diagnostics on a hard instance).
- Documentation page "Concurrent sweeps and error control", including a
  **Reproducibility** section: `Zipper` draws its randomized range-finder sketch
  from the global RNG, so `sweep_transformations` seeds per task while a bare
  `low_energy_spectrum` leaves seeding to the caller.

### Changed

- `low_energy_spectrum` gained `show_progress` (default `true`; set `false` when
  solving concurrently, since interleaved progress bars are unreadable),
  `schmidt_spectrum`, and `retain_mps` keyword arguments.
- **Breaking (minor):** `low_energy_spectrum` no longer computes the per-row
  Schmidt spectra by default — its second return value is empty unless
  `schmidt_spectrum = true`. Computing them costs an untruncated CPU SVD per site
  per row, and no caller in this repository (or in the published examples) reads
  them.
- `variational_compress!` logs its per-sweep convergence trace at `@debug`
  instead of `@info`. It fires once per sweep per row, so at info level it buried
  everything else a caller printed.

### Performance

- **The CPU/GPU crossover is now established.** Solving identical configurations on
  each device across three instance sizes, four bond dimensions and both sparsity
  modes (14 cells, energies agreeing everywhere): the host path is ~45× faster at 36
  spins, 1.5× faster at 2048 spins and bond 16, and loses only once both the
  instance and the bond dimension are large — 2048 spins at bond 32, where the
  device wins by 3% (`Dense`) and 17% (`Sparse`). `MpsContractor` still defaults to
  `onGPU = true`, which is right for the D-Wave-scale sparse regime the package
  targets and wrong for exploratory work; measure before assuming.

- **The solver is host-bound at every size measured**, which is the finding behind
  the two allocation changes below. GPU utilization is 5.9% on a 128-spin solve and
  12.8% at 2048 spins, while host-side CUDA API calls account for only 21–25% of
  wall time; two thirds is host-side Julia work, of which garbage collection alone
  is 18% (GPU) to 33% (CPU). Batching kernels — the intuitive response to an idle
  device — was therefore not pursued: eliminating the entire CUDA API share caps at
  ≈1.3×, well below what reducing allocation returned.

- **Contraction temporaries no longer go on the collected heap.** The seven
  multi-tensor contractions in `contractions/dense.jl` now use TensorOperations'
  `ManualAllocator`, selected per call so device arrays keep the default (it returns
  host memory). On a 2048-spin bond-32 CPU solve: **1.25×** (24.2 → 19.4 s), with
  allocated bytes down 65% (90.9 → 32.0 GiB) and GC from 7.7 s to 4.2 s. Neutral on
  GPU by construction. In isolation the allocator is ~7% *slower* per call, so the
  gain is entirely the collection it avoids — it had to be measured in a full solve.


- **`branch_states` no longer materializes the expanded state set twice.** It built
  a 3-D temporary holding every branched configuration and then copied it again via
  `collect.(eachcol(...))`, on top of a `reduce(hcat, ...)` of the input; the output
  vectors are now written directly. Profiling a bond-32 `Zipper` solve attributed
  **52.7% of all allocated bytes** to that single line, making it by far the largest
  allocation source in a solve — larger than anything in the tensor kernels.

  Measured with alternating separate-process paired runs: allocations −22.4% on GPU
  (0.735 → 0.570 GiB per solve) and −12.0% on CPU (1.356 → 1.194 GiB), for a wall
  time gain of ~1.13× on CPU (consistent across pairs) and ~1.05–1.09× on GPU
  (noisier). Energies are unchanged, and `test/engine/branch_states.jl` pins the
  expansion contract — in particular the ordering, which callers rely on to pair
  states index-for-index with energies and probabilities.

- **The branched state set is now backed by one matrix instead of one vector per
  state.** A second pass at the same site: `branch_states_view` (new, internal)
  writes the expansion into a single `Matrix{Int}` and returns column views, which
  `branch_solution` uses on the hot path. At 2048 spins / bond 32 this is
  **1.157× on GPU** (26.6 → 23.0 s) and 1.063× on CPU, with GC down **36%**
  (4.97 → 3.16 s) — while allocated *bytes* fall only 4%. That gap is the point:
  garbage-collection cost tracks the number of objects, not their volume, and the
  old form created tens of thousands of small vectors per call.

  `Solution.states`/`spins` and the ~11 dispatch points that consume a state were
  widened to `AbstractVector{Int}` accordingly. `boundary_states` still returns
  freshly built `Vector{Int}`, so the `∂v` path — every geometry's
  `conditional_probability`, `branch_probability` — was unaffected. The public
  `Solution` also still carries `Vector{Int}` states, since `low_energy_spectrum`
  materializes them in its final permutation; views never escape the search.

  `branch_states` itself keeps returning `Vector{Vector{Int}}`: it is exported and
  `test/engine/api_compat.jl` pins that element type, so the fast path is a
  separate internal entry point rather than a change to the published contract.

### Fixed

- `empty_solution` built its empty configurations as `fill(Vector{Int}[], n)`,
  producing a `Vector{Vector{Vector{Int}}}` that type-checked only because
  converting an *empty* `Vector{Vector{Int}}` to `Vector{Int}` succeeds
  elementwise by vacuity. Harmless in practice but latent; now written explicitly.

- `kernel_batch_size` now honours an explicit device-memory budget when one is in
  scope, instead of always sizing against total free device memory. Concurrent
  solves previously each measured the whole free pool.


Breaking release. The four separately-registered packages that made up the stack
(`SpinGlassTensors`, `SpinGlassNetworks`, `SpinGlassExhaustive`, `SpinGlassEngine`)
are now internal modules of a single installable `SpinGlassPEPS` package.

- `corner_matrix` for `SiteTensor` (threw on every input) and for `VirtualTensor`
  (scrambled trailing dimensions).
- Exhaustive-search GPU kernel launched too few blocks, silently truncating the
  spectrum for problems larger than ~9 spins.
- Cross-package name collisions (`energy`, `bond_energy`, `interaction_energy`,
  `local_energy`, `projector`) that raised `UndefVarError` on Julia ≥ 1.12.
- Numerous latent bugs surfaced by the audit (undefined-variable branches,
  unthrown `ArgumentError`s, an `abs`-of-`Bool`, phantom exports, dead files).

### Changed (breaking)

- **Single package.** `pkg> add SpinGlassPEPS` installs the entire solver stack.
  The former sub-packages live as internal modules under `src/{tensors,networks,
  exhaustive,engine}`, so ordinary use is unchanged: `using SpinGlassPEPS`.
  Advanced users can still qualify a module explicitly, e.g.
  `SpinGlassPEPS.SpinGlassNetworks`.
- **Curated public API.** `using SpinGlassPEPS` now brings a curated public
  surface (~65 symbols: model construction, geometries/sparsity/layouts,
  `PEPSNetwork`/`MpsContractor`/parameters, strategies, transformations, the
  search + droplet API, exhaustive + QUBO helpers, belief-propagation reduction,
  the documented core types, and the compatibility shims) instead of
  re-exporting all ~240 symbols of the internal stack. Low-level kernels,
  accessors, abstract types, and helpers are now internal — still fully usable
  via the submodules (`using SpinGlassPEPS.SpinGlassEngine`, …), just no longer
  part of the top-level contract. The stand-alone `SpinGlassTensors`,
  `SpinGlassNetworks`, `SpinGlassExhaustive`, and `SpinGlassEngine` packages are
  no longer used or released.
- **Potts Hamiltonian is a concrete type.** The clustered Hamiltonian consumed by
  the solver is now a typed `PottsHamiltonian` struct rather than a
  `MetaGraphs` `Dict{Symbol,Any}` property bag. Data is read through accessor
  functions (`interaction`, `left_projector`, `right_projector`, `cluster_spectrum`,
  `projector_pool`, …). The legacy `LabelledGraph` representation still backs the
  two-site belief-propagation graph and the RMF loader and is accepted anywhere a
  Hamiltonian is expected (`PottsLike`).
- **Contraction caching is contractor-owned.** Process-global `Memoization`
  caches are replaced by a `ContractionCache` held on each `MpsContractor`, with
  explicit per-row eviction. `Memoization` is no longer a dependency.
- **Concrete network/contractor fields.** `PEPSNetwork` and `MpsContractor` carry
  concrete type parameters for the Hamiltonian and network; `vertex_map` is a
  concrete `VertexMap` rather than an untyped `Function`.

### Deprecated

- `merge_branches(...; merge_type=…, update_droplets=…)` — use `merge_prob` and
  `droplets_encoding`; the old keywords warn and forward.
- `clear_memoize_cache` / `clear_memoize_cache_after_row` are now no-op shims kept
  for API compatibility.

### Removed

- The forced `MetaGraphs`-property-bag storage on the Potts side (see Changed).
- Dead code, stale generated docs, and unused dependencies identified during the
  audit.

## [1.5.0] and earlier

Released as the umbrella package that depended on the four separate
`SpinGlass*` packages. See the git history of those packages for details.
