# Changelog

All notable changes to `SpinGlassPEPS.jl` are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [2.0.0]

Breaking release. The four separately-registered packages that made up the stack
(`SpinGlassTensors`, `SpinGlassNetworks`, `SpinGlassExhaustive`, `SpinGlassEngine`)
are now internal modules of a single installable `SpinGlassPEPS` package.

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

### Added

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

### Fixed

- `corner_matrix` for `SiteTensor` (threw on every input) and for `VirtualTensor`
  (scrambled trailing dimensions).
- Exhaustive-search GPU kernel launched too few blocks, silently truncating the
  spectrum for problems larger than ~9 spins.
- Cross-package name collisions (`energy`, `bond_energy`, `interaction_energy`,
  `local_energy`, `projector`) that raised `UndefVarError` on Julia ≥ 1.12.
- Numerous latent bugs surfaced by the audit (undefined-variable branches,
  unthrown `ArgumentError`s, an `abs`-of-`Bool`, phantom exports, dead files).

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
