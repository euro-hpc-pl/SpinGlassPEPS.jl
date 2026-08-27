# SpinGlassPEPS stack — refactor plan

Audit date: 2026-07-11. Scope: SpinGlassTensors, SpinGlassNetworks, SpinGlassEngine,
SpinGlassExhaustive, SpinGlassPEPS (umbrella), LabelledGraphs (~11k LOC of Julia), checked
against the published description (arXiv:2502.02317, SoftwareX 31, 102257 (2025)).

Architecture update (2026-07-13): the four SpinGlass implementation packages have
been consolidated into one installable `SpinGlassPEPS` package. Their module
boundaries remain under `src/tensors`, `src/networks`, `src/exhaustive`, and
`src/engine`; historical package and `lib/` paths below describe the pre-consolidation
state unless a phase explicitly says otherwise.

## Verdict

A fundamental refactor is justified. It will improve performance — but the wins come from a
handful of specific, verifiable defects (mostly GPU transfer churn and a global-cache design
that blocks parallelism), not from structural cleanliness itself. The cleanliness work buys
maintainability and the *ability* to fix those defects safely. Precondition for all
performance claims: a benchmark harness — none currently works (`publication/bench.jl`
points at instance directories that do not exist).

## Diagnosis

### Disease 1 — global mutable state as the caching architecture

~20 functions in Engine are `@memoize Dict` (boundary MPS, MPO layers, environments), keyed
on the whole mutable contractor plus boundary slices (`∂v[2:l]` — O(L²) allocation per
boundary, vector hashing per lookup). The caches are process-global, thread-unsafe, hold GPU
arrays, and are invalidated by manual surgery scattered through the search loop:
`clear_memoize_cache_after_row()`, `Memoization.empty_cache!(SpinGlassTensors.sparse)`
(Engine reaching into a sibling package's cache, `search.jl:571`), `empty!(ctr.peps.lp, :GPU)`.
Replacing this with a contractor-owned cache is the single change that unlocks parallelizing
the 8-lattice-transform sweep — the cheapest large end-to-end speedup available.

### Disease 2 — the type system is fought, not used

- The central domain object (Potts Hamiltonian) is a `LabelledGraph` over a MetaGraphs
  `Dict{Symbol,Any}` property bag (`:en`, `:spectrum`, `:ipl/:ipr`, `:pool_of_projectors`).
  Every access in the search loop is dynamically typed.
- `PEPSNetwork` carries `vertex_map::Function` and an unparametrized `LabelledGraph`
  (`PEPS.jl:50-60`); `MpsContractor` carries `statistics::Any` that grows forever and is
  never read (`contractor.jl:177`).
- `QMps`/`QMpo` store tensors as `Dict{Site, Union{Tensor{T,2},Tensor{T,3},Tensor{T,4}}}`
  with `Site = Union{Int,Rational{Int}}` (`mps/base.jl`).
- Meanwhile user-facing type parameters explode: `PEPSNetwork{Lattice{Layout},Sparsity,T}` ×
  `MpsContractor{Strategy,Gauge,T}` × 8 transforms, with parameter letters that swap meaning
  between files. Types are heavy where they don't pay (API combinatorics, compile time) and
  absent where they would (hot struct fields).

### Disease 3 — six repos in version lockstep (resolved)

Engine's compat holds Networks/Tensors at current minors, so every release is a hand-cascaded
train of "Update Project.toml" commits. The umbrella is 6 lines of `@reexport`; its test
suite re-runs three subpackage suites and skips SpinGlassExhaustive. Layering is inverted:
Networks depends on Tensors only for `PoolOfProjectors` (a domain concept); Engine
hard-depends on SpinGlassExhaustive it never imports. Docs are aggregated by regex-scraping
and `eval`ing sibling repos' `make.jl`, with 63 stale generated files tracked in git. CI is
PR-only (master never tested), single self-hosted GPU runner, single Julia version, with
`onGPU = true` hardcoded — the CPU path has no CI coverage anywhere.

### Device handling

Three coexisting idioms (`onGPU::Bool`, `:CPU`/`:GPU` Symbols, `typeof(x) <: CuArray`);
CUDA/cuTENSOR/MKL as hard deps everywhere (MKL forced on all users; Networks loads CUDA and
MKL it never uses); `CUDA.allowscalar(false)` mutated globally at module load. The kernels
themselves are already array-generic — this is plumbing unification, not a rewrite.

### Verified performance defects (all confirmed in code)

| # | Defect | Evidence | Expected effect of fix |
|---|--------|----------|------------------------|
| 1 | Every QR/RQ runs on CPU (`qr(Array(M))` + `CuArray.(…)` back), every site, every sweep | `SpinGlassTensors/src/linear_algebra_ext.jl:33` | Largest single win; boundary-MPS preprocessing is sweep-dominated — plausibly 1.5–3× that phase on GPU |
| 2 | `right_env` re-uploads and downloads at every recursion level; zipper matvecs cross PCIe twice per iteration; `conditional_probability` re-uploads per call | `SpinGlassEngine/src/contractor.jl:585-611`, `SpinGlassTensors/src/zipper.jl:251-272` | Thousands of small synchronous transfers per row eliminated (latency-bound) |
| 3 | VirtualTensor kernels recompute projector merges (rank_reveal + H2D uploads) per call; the memoize that fixed it is commented out | `SpinGlassTensors/src/contractions/virtual.jl:4,66-85` | Hot-path win for Sparse (Pegasus/Zephyr) workloads |
| 4 | KingSingleNode inner kernel downloads to CPU and runs a serial scalar loop (authors' own `# REWRITE` comment) | `SpinGlassEngine/src/king_single_node.jl:214-221`; fixed pattern exists at `square_cross_double_node.jl:443-456` | Serial CPU → batched GEMM on device |
| 5 | Global memoize caches (see Disease 1) | `contractor.jl:279-707`, `search.jl:567-623` | Modest direct win; big indirect: bounded GPU memory + thread-safety → near-linear speedup over 8 transforms |
| 6 | Hardcoded memory budgets: 2³²/2³³ bytes, 8 bytes/element (wrong for Float32, ignores device) | `contractions/site.jl:25,132`, `square_cross_double_node.jl:439` | Correct batching on any GPU/precision |
| 7 | `add_projector!` dedupes by linear scan — O(n²) construction | `SpinGlassTensors/src/projectors.jl:129-136` | Faster network construction on large instances |

What will **not** get faster: the dense contraction kernels themselves (BLAS/cuTENSOR-bound).
CPU-only runs improve less (threading, king kernel, allocation churn). TTFX likely improves
from shrinking the parametric explosion — hypothesis, not measurement.

### Shipped-broken list (fix regardless)

- Dead, never-included 363-line `SpinGlassEngine/src/network_tensors.jl` referencing
  nonexistent fields and an undefined variable.
- Nine phantom exports: Networks `pegasus_lattice_masoud`, `pegasus_lattice_tomek`, `nodes`
  (+ accidental `rank` re-export); Engine `mps_top_approx`, `update_gauges_with_balancing!`,
  `decode_to_spin`; Exhaustive exports of commented-out functions.
- `truncate!` `:left` branch references undefined `args` (`mps/canonise.jl:39`);
  `is_consistent` error message uses undefined `i` (`mps/utils.jl:16`); `Base.Array(CM)` for
  the zipper adjoint references nonexistent field `.ten` (`zipper.jl:209`);
  keyword default `iter = iter` is a guaranteed UndefVarError (`Networks/src/truncate.jl:197`);
  `abs(n[2] - n[4] == 2)` applies `abs` to a Bool (`Networks/src/utils.jl:141`);
  two `ArgumentError`s constructed but never thrown (`Engine/src/operations.jl:78,140`).
- Exhaustive's main test compares a value to itself (`test/ising.jl:10`);
  `exhaustive_search` launches ~N threads for 2^N states (`src/ising.jl:151-158`).
- Dead code: `virtual.jl:8-64` unused projector helpers with *inverted* CPU/GPU branches;
  commented-out CUDA method copies; ~140 commented-out `@cast` lines; dead lattice functions
  calling undefined helpers (`Networks/src/lattice.jl:97-125`); umbrella `benchmarks/` uses
  pre-2021 APIs (LightGraphs) and cannot run.
- The API published in the SoftwareX paper already throws on current releases: renames
  (`merge_type` → `merge_prob`, positional `SingleLayerDroplets`) shipped in 1.x minors with
  zero `@deprecate` anywhere.

## Target architecture

One registered package containing four internal modules. `pkg> add SpinGlassPEPS`
installs the complete stack, while the internal module boundaries preserve the existing
Tensors, Networks, Exhaustive, and Engine namespaces. The consolidation is complete;
the longer-term target still includes two package extensions:

- **Core** — `PoolOfProjectors` (+ merged-projector registry), `rank_reveal`, device traits
  over `Adapt.adapt`. Fixes the inverted layering (Networks currently imports the entire
  kernel package for one domain type).
- **Tensors** — QMps/QMpo and sparse formats parametrized on concrete storage type (array
  type = device truth; `onGPU`/Symbols/`move_to_CPU!` become `adapt`); kernels take an
  explicit workspace owning the projector-sparse-matrix cache (replaces the pirated,
  memoized `SparseArrays.sparse`).
- **Models** — a real `PottsHamiltonian{T}` struct (typed spectra, interaction matrices,
  projector indices) replacing the MetaGraphs property bag; MetaGraphs deleted.
- **Engine** — explicit, documented `AbstractGeometry` protocol with a conformance testset
  (replaces four copy-pasted geometry files, ~2,200 lines re-implementing one implicit
  ~10-function interface); `MpsContractor` owns a typed `ContractionCache` with `drop_row!`
  eviction, replacing every `@memoize` global.
- **ext/CUDAExt** — CUSOLVER qr/rq, CuSparse builders, GPU brute force (absorbs
  SpinGlassExhaustive), no load-time global mutation. **ext/FileFormatsExt** — HDF5/CSV.
  MKL, Memoization, MetaGraphs, JLD2 removed outright.
- One curated export list (~35 names instead of ~240 across per-file export blocks), with
  `deprecations.jl` preserving every symbol in the paper's listings (`clear_memoize_cache`
  becomes a no-op shim; `onGPU` kwarg maps to a device argument).

LabelledGraphs stays a separate leaf (two small type-stability fixes → 0.5). SpinGlassMPS
and SpinGlassDynamics are archived.

## Migration plan (tests green at the end of every phase)

- [x] **Phase 0 — mechanics, zero code change.** Consolidated package with one root
  `Project.toml`; CI on push+PR with a GitHub-hosted CPU
  leg (`onGPU = CUDA.functional()`) plus the self-hosted GPU leg; benchmark harness
  (`benchmark/`) over reference instances recording wall time by phase, peak memory,
  allocations, TTFX, energy-vs-oracle correctness. Record baselines before any code change.
- [x] **Phase 1 — deletion and latent bugs.** Dead file(s), phantom exports, commented-out
  blocks, dead `benchmarks/`, stale generated docs, unearned dependency edges
  (Engine→Exhaustive to test-only; Networks drops CUDA/MKL; Exhaustive drops Tensors;
  umbrella drops Documenter from runtime deps), the latent-bug list above, vacuous/empty/
  orphaned tests. Nothing removed is tested, so this is trivially green.
- [x] **Phase 2 — kernel test net.** DONE: unit tests for all five kernel families against
  independent dense references (~1,400 new assertion lines, CPU+GPU, Float64+Float32, all
  size-heuristic branches). Writing them surfaced two real bugs, both fixed: SiteTensor
  `corner_matrix` threw for every input (adjoint of a 3-D array); VirtualTensor
  `corner_matrix` returned scrambled trailing dimensions. The `mwe*.jl` harnesses were
  re-run in full (320 configurations, repeats, negative control): **zero historical
  probability-vs-Boltzmann failures reproduce** on the current stack; the properties are
  pinned as plain regression tests in `test/known_failures.jl` instead of `@test_broken`.
- [x] **Phase 3 — GPU quick wins.** DONE, with measured corrections to the audit's
  predictions (all on RTX 5080, min-of-3 warm solves, alternating A/B where noise demanded):
  - Device-native QR/RQ via CUSOLVER: **landed, gated at 2^15 elements** — below that CPU
    LAPACK + PCIe wins 10x; above it CUSOLVER wins 1.6–5.7x. Net ~10–20% at bond 32.
  - Device-resident `right_env` cache: **deferred to Phase 4.** A/B showed the GC/pool
    pressure of CuArray values in the memoize dict outweighs the PCIe traffic at small
    bond dimensions (gc_time 2.4x, patho +20%). Needs the explicit cache's deterministic
    freeing. The allocation-free `maximum(abs, x)` norms were kept.
  - Projector-merge registry in `PoolOfProjectors` (was recomputed per VirtualTensor
    kernel call): landed.
  - King-node conditional probability: serial CPU scalar loop replaced by device-resident
    batched GEMM: landed.
  - `kernel_batch_size(T, per_item, onGPU)` replaces hardcoded 2^32/2^33-byte, 8-byte/element
    budgets (halved Float32 batches; overflowed small GPUs): landed.
  - Zipper matvec PCIe crossings: **still open** — coupled to LowRankApprox's CPU Krylov
    vectors; needs a device-side randomized SVD (Phase 4+ follow-up).
- [x] **Phase 4 — kill global state.** DONE (core): every `@memoize` in the stack replaced
  by contractor-owned `ContractionCache{S}` with explicit row eviction and solve-end
  release; `SparseArrays.sparse` piracy deleted (`projector_matrix` cached in
  `PoolOfProjectors`); Memoization.jl removed from all dependencies;
  `clear_memoize_cache*` kept as documented no-op shims. Full Engine suite + umbrella
  green; benchmark: faster than memoize on every case (up to −5.9% time, −6.5% allocs —
  no more hashing contractor+vector keys per lookup). Still open as follow-ups:
  device-resident environments with deterministic freeing, threaded 8-transform sweep,
  GPU-side randomized SVD for the zipper.
- [x] **Phase 5 step 1 — accessor layer.** All 67 property reads across the stack go
  through 12 accessor functions (`accessors.jl`); the MetaGraphs property bag is now an
  implementation detail with a 12-function surface.
- [x] **Phase 5 — typed model layer.** DONE for the standard (solver-facing) flavor:
  `PottsHamiltonian{L,T,C,SP,OE}` with concrete typed fields (clusters, spectra,
  `ClusterInteraction{T}` edge data, owned projector pool) over a SimpleDiGraph topology;
  parametric `Spectrum{T,S}`/`MergedEnergy`; openGM loader eltype fixed; accessor layer
  + MetaGraphs-compat `get_prop` shims keep old call sites working. The legacy
  representation still backs the 2-site BP graph and RMF loader behind the same
  accessors (`PottsLike`); MetaGraphs is still a dep for IsingGraph — full removal rides
  with a later IsingGraph typing pass. Full suite + umbrella green; benchmark
  neutral-to-positive (sparse −5.7%, allocations down across cases). `PottsHamiltonian{T}` with accessors mirroring today's
  `get_prop` call sites; migrate Networks internals, then Engine call sites; drop MetaGraphs;
  edge-driven belief propagation rewrite.
- [x] **Phase 6 (core) — Engine type system.** Concrete hot-struct fields landed:
  `VertexMap` callable replaces the `Function` field, `PEPSNetwork{T,S,R,PH}` and
  `MpsContractor{T,R,S,N}` carry concrete Hamiltonian/network types (3-param public
  constructors preserved), concrete pool/gauges fields; geometry protocol documented in
  `geometry.jl` and enforced by a 110-assertion conformance testset; droplet tests
  memory-gated (skip < 12 GiB with a warning — the suite OOMs 3080-class cards).
  Benchmark (this slice alone): −4.5% to −10% time, allocations to −10.7%.
  Still open: merging the four `conditional_probability` bodies over a shared skeleton,
  and replacing the per-build `Val(Symbol)` tensor-species dispatch with a typed spec
  table — both want a dedicated session. Concrete fields, standardized parameter order,
  geometry protocol + shared `conditional_probability` skeleton (one geometry at a time,
  `SquareSingleNode` first), tensor species resolved at network construction instead of
  `Val(Symbol)` per build.
- [~] **Phase 7 (compat slice) — API 2.0.** The published SoftwareX listings run again:
  `merge_branches(...; merge_type=..., update_droplets=...)` deprecation shims and the
  positional `SingleLayerDroplets` constructor. Still open for the 2.0 release: curated
  export lists (~240 → ~35, needs maintainer sign-off on the public surface), one
  Documenter build replacing the regex-scrape aggregation, version bumps and registered
  releases. Curated exports, `deprecations.jl` covering the paper listings,
  one Documenter build replacing the regex-scrape aggregation, and release SpinGlassPEPS 2.0
  (breaking changes only at majors, preceded by one deprecation minor).

## Notes

- The audit's per-package reports (with full file:line evidence) were produced by a
  multi-agent read of every source file; the headline claims in this document were
  re-verified by hand against the working tree on 2026-07-11.
- Historical scratch from the former Engine repository documented known
  probability-vs-Boltzmann mismatches across the config matrix — convert to
  `@test_broken` sets in Phase 2 rather than losing that information.
